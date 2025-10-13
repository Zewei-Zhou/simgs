#
# Copyright (C) 2023, Inria
# GRAPHDECO research group, https://team.inria.fr/graphdeco
# All rights reserved.
#
# This software is free for non-commercial, research and evaluation use 
# under the terms of the LICENSE.md file.
#
# For inquiries contact  george.drettakis@inria.fr
#
import torch
import math
import gsplat
from gsplat.cuda._wrapper import fully_fused_projection, fully_fused_projection_2dgs
import torch.nn.functional as F
# from gsplat.cuda._torch_impl import _fully_fused_projection as fully_fused_projection
# from gsplat.cuda._torch_impl_2dgs import _fully_fused_projection_2dgs as fully_fused_projection_2dgs

def render(viewpoint_camera, pc, pipe, bg_color, visible_mask=None, training=True, object_mask=None):
    """
    Render the scene. 
    
    Background tensor (bg_color) must be on GPU!
    """
    if pc.explicit_gs:
        xyz, color, opacity, scaling, rot, sh_degree, selection_mask = pc.generate_explicit_gaussians(visible_mask)
    else:
        if object_mask is None:
            xyz, offset, color, opacity, scaling, rot, sh_degree, selection_mask, semantics = pc.generate_neural_gaussians(viewpoint_camera, visible_mask, training)
        else:
            xyz, offset, color, opacity, scaling, rot, sh_degree, selection_mask, semantics = pc.generate_neural_gaussians(viewpoint_camera, visible_mask & object_mask, training)

    # Set up rasterization configuration
    K = torch.tensor([
            [viewpoint_camera.fx, 0, viewpoint_camera.cx],
            [0, viewpoint_camera.fy, viewpoint_camera.cy],
            [0, 0, 1],
        ],dtype=torch.float32, device="cuda")
    viewmat = viewpoint_camera.world_view_transform.transpose(0, 1) # [4, 4]
    
    if pc.gs_attr == "3D":
        render_colors, render_alphas, render_semantics, info = gsplat.rasterization(
            means=xyz,  # [N, 3]
            quats=rot,  # [N, 4]
            scales=scaling,  # [N, 3]
            opacities=opacity.squeeze(-1),  # [N,]
            colors=color,
            viewmats=viewmat[None],  # [1, 4, 4]
            Ks=K[None],  # [1, 3, 3]
            width=int(viewpoint_camera.image_width),
            height=int(viewpoint_camera.image_height),            
            backgrounds=bg_color[None],
            packed=False,
            sh_degree=sh_degree,
            render_mode=pc.render_mode,
            features=semantics.detach(),
            tile_size=getattr(pipe, 'tile_size', 16)
        )
    elif pc.gs_attr == "2D":
        (render_colors, 
        render_alphas,
        render_normals,
        render_normals_from_depth,
        render_distort,
        render_median,
        render_semantics,
        info) = \
        gsplat.rasterization_2dgs(
            means=xyz,  # [N, 3]
            quats=rot,  # [N, 4]
            scales=scaling,  # [N, 3]
            opacities=opacity.squeeze(-1),  # [N,]
            colors=color,
            viewmats=viewmat[None],  # [1, 4, 4]
            Ks=K[None],  # [1, 3, 3]
            width=int(viewpoint_camera.image_width),
            height=int(viewpoint_camera.image_height),            
            backgrounds=bg_color[None] if pc.render_mode not in ["RGB+D", "RGB+ED"] \
                else torch.cat((bg_color[None], torch.zeros((1, 1), device="cuda")), dim=-1),
            packed=False,
            sh_degree=sh_degree,
            render_mode=pc.render_mode,
            features=semantics.detach(),
            tile_size=getattr(pipe, 'tile_size', 16)
        )
    else:
        raise ValueError(f"Unknown gs_attr: {pc.gs_attr}")

    # [1, H, W, 3] -> [3, H, W]
    if render_colors.shape[-1] == 4:
        colors, depths = render_colors[..., 0:3], render_colors[..., 3:4]
        depth = depths[0].permute(2, 0, 1)
    else:
        colors = render_colors
        depth = None

    rendered_image = colors[0].permute(2, 0, 1)
    radii = info["radii"].squeeze(0) # [N,]
    try:
        info["means2d"].retain_grad() # [1, N, 2]
    except:
        pass

    render_alphas = render_alphas[0].permute(2, 0, 1)

    # Those Gaussians that were frustum culled or had a radius of 0 were not visible.
    return_dict = {
        "render": rendered_image,
        "scaling": scaling,
        "viewspace_points": info["means2d"],
        "visibility_filter" : radii > 0,
        "visible_mask": visible_mask,
        "selection_mask": selection_mask,
        "opacity": opacity,
        "render_depth": depth,
        "radii": radii,
        "render_alphas": render_alphas,
        "render_semantics": render_semantics,
    }
    
    if pc.gs_attr == "2D":
        return_dict.update({
            "render_normals": render_normals,
            "render_normals_from_depth": render_normals_from_depth,
            "render_distort": render_distort,
        })

    return return_dict

def prefilter_voxel(viewpoint_camera, pc):
    """
    Render the scene. 
    
    Background tensor (bg_color) must be on GPU!
    """
    means = pc.get_anchor[pc._anchor_mask]
    scales = pc.get_scaling[pc._anchor_mask][:, :3]
    quats = pc.get_rotation[pc._anchor_mask]
    
    # Set up rasterization configuration
    Ks = torch.tensor([
            [viewpoint_camera.fx, 0, viewpoint_camera.cx],
            [0, viewpoint_camera.fy, viewpoint_camera.cy],
            [0, 0, 1],
        ],dtype=torch.float32, device="cuda")[None]
    viewmats = viewpoint_camera.world_view_transform.transpose(0, 1)[None]

    N = means.shape[0]
    C = viewmats.shape[0]
    device = means.device
    assert means.shape == (N, 3), means.shape
    assert quats.shape == (N, 4), quats.shape
    assert scales.shape == (N, 3), scales.shape
    assert viewmats.shape == (C, 4, 4), viewmats.shape
    assert Ks.shape == (C, 3, 3), Ks.shape

    # Project Gaussians to 2D. Directly pass in {quats, scales} is faster than precomputing covars.
    if pc.gs_attr == "3D":
        proj_results = fully_fused_projection(
            means,
            None,  # covars,
            quats,
            scales,
            viewmats,
            Ks,
            int(viewpoint_camera.image_width),
            int(viewpoint_camera.image_height),
            eps2d=0.3,
            packed=False,
            near_plane=0.01,
            far_plane=1e10,
            radius_clip=0.0,
            sparse_grad=False,
            calc_compensations=False,
        )
    elif pc.gs_attr == "2D":
        # densifications = (
        #     torch.zeros((C, N, 2), dtype=means.dtype, device="cuda")
        # )
        # Project Gaussians to 2D. Directly pass in {quats, scales} is faster than precomputing covars.
        proj_results = fully_fused_projection_2dgs(
            means,
            quats,
            scales,
            viewmats,
            Ks,
            int(viewpoint_camera.image_width),
            int(viewpoint_camera.image_height),
            eps2d=0.3,
            packed=False,
            near_plane=0.01,
            far_plane=1e10,
            radius_clip=0.0,
            sparse_grad=False,
        )

        # torch implementation
        # proj_results = fully_fused_projection_2dgs(
        #     means,
        #     quats,
        #     scales,
        #     viewmats,
        #     Ks,
        #     int(viewpoint_camera.image_width),
        #     int(viewpoint_camera.image_height),
        #     near_plane=0.01,
        #     far_plane=1e10,
        #     eps=0.3,
        # )
    else:
        raise ValueError(f"Unknown gs_attr: {pc.gs_attr}")
    
    # The results are with shape [C, N, ...]. Only the elements with radii > 0 are valid.
    radii, means2d, depths, conics, compensations = proj_results
    # radii, means2d, depths, M, normals = proj_results # torch impl
    camera_ids, gaussian_ids = None, None
    
    visible_mask = pc._anchor_mask.clone()
    visible_mask[pc._anchor_mask] = radii.squeeze(0) > 0

    return visible_mask

def render_with_sky(viewpoint_cam, gaussians, pipe, bg_color, visible_mask=None, training=True, object_mask=None, sky_model= None):
    """Integrate sky model into rendering pipeline"""
    render_pkg = render(viewpoint_cam, gaussians, pipe, bg_color, visible_mask, training, object_mask)
    if sky_model is not None:
        H, W = viewpoint_cam.image_height, viewpoint_cam.image_width
        sky_colors = generate_sky_colors(viewpoint_cam, sky_model, H, W, training)
        rendered_image = render_pkg["render"]
        rendered_opacity = render_pkg["render_alphas"]  # [1, H, W] from render function
        T_bg = (1.0 - rendered_opacity).clamp(0.0, 1.0)  # [1, H, W]
        
        # Cf = C + (1-O)fsky(d)

        if hasattr(viewpoint_cam, 'sky_mask') and viewpoint_cam.sky_mask is not None:
            sky_mask = viewpoint_cam.sky_mask.float()  # Original sky mask [1, H, W]
            sky_mask_4d = sky_mask.unsqueeze(0)  # [1, 1, H, W]
            inv = 1.0 - sky_mask_4d  # [1, 1, H, W]
            inv_eroded = F.max_pool2d(inv, kernel_size=5, stride=1, padding=2)
            sky_mask_eroded = 1.0 - inv_eroded  # [1, 1, H, W]
            sky_mask_soft = sky_mask_eroded.squeeze(0)  # [1, H, W] - same shape as T_bg

        else:
            # Fallback: use opacity-based sky detection
            tau, sharp = 0.2, 10.0                                      
            sky_mask_soft = torch.sigmoid((T_bg - tau) * sharp)  # Same shape as T_bg
        
        gate = T_bg * sky_mask_soft
        final_image = rendered_image + gate * sky_colors
        render_pkg["render"] = final_image
        render_pkg["sky_colors"] = sky_colors
    return render_pkg

def generate_sky_colors(viewpoint_cam, sky_model, H, W, training):
    """render sky colors for given pixels"""

    i, j = torch.meshgrid(
        torch.arange(W, dtype=torch.float32, device="cuda"),
        torch.arange(H, dtype=torch.float32, device="cuda"),
        indexing="xy"
    )
    
    if training:
        i = i + torch.rand_like(i) - 0.5
        j = j + torch.rand_like(j) - 0.5
    
    ndc_x = (i - viewpoint_cam.cx) / viewpoint_cam.fx
    ndc_y = (j - viewpoint_cam.cy) / viewpoint_cam.fy
    directions_cam = torch.stack([ndc_x, ndc_y, torch.ones_like(ndc_x)], dim=-1)
    
    c2w = torch.inverse(viewpoint_cam.world_view_transform.T)
    directions_world = (directions_cam @ c2w[:3, :3].T)
    directions_world = directions_world / torch.norm(directions_world, dim=-1, keepdim=True)
    
    # Ensure directions are contiguous
    directions_world = directions_world.contiguous()
    
    try:
        sky_colors = sky_model(directions_world)  # [H, W, 3]
    except Exception as e:
        print(f"Warning: Sky model forward pass failed ({e}), using fallback")
        # Use the same device as directions_world for fallback
        sky_colors = torch.ones(H, W, 3, device=directions_world.device, dtype=torch.float32) * 0.5
    
    return sky_colors.permute(2, 0, 1)  # [3, H, W]