import torch
from scene import Scene
import os
import sys
import yaml
from tqdm import tqdm
from os import makedirs
import torchvision
from argparse import ArgumentParser
from utils.mesh_utils import GaussianExtractor, to_cam_open3d, post_process_mesh
from utils.general_utils import parse_cfg
import open3d as o3d
from datetime import datetime

if __name__ == "__main__":
    # Set up command line argument parser
    parser = ArgumentParser(description="Testing script parameters")
    parser.add_argument('-m', '--model_path', type=str, required=True)
    parser.add_argument("--scene_name", default=None)
    parser.add_argument("--iteration", default=-1, type=int)
    parser.add_argument("--voxel_size", default=-1.0, type=float, help='Mesh: voxel size for TSDF')
    parser.add_argument("--depth_trunc", default=-1.0, type=float, help='Mesh: Max depth range for TSDF')
    parser.add_argument("--sdf_trunc", default=-1.0, type=float, help='Mesh: truncation value for TSDF')
    parser.add_argument("--num_cluster", default=10, type=int, help='Mesh: number of connected clusters to export')
    parser.add_argument("--query_label_id", default=-1, type=int, help='Mesh: id of queried gaussians. Use -1 for all objects together, -2 for all individual objects separately')
    parser.add_argument("--unbounded", action="store_true", help='Mesh: using unbounded mode for meshing')
    parser.add_argument("--mesh_res", default=2048, type=int, help='Mesh: resolution for unbounded mesh extraction')
    args = parser.parse_args(sys.argv[1:])
    
    with open(os.path.join(args.model_path, "config.yaml")) as f:
        cfg = yaml.load(f, Loader=yaml.FullLoader)

    args.scene_name = args.model_path.split('/')[-2]

    if args.scene_name is not None:
        try:
            cfg["model_params"]["exp_name"] = os.path.join(cfg["model_params"]["exp_name"], args.scene_name)
            cfg["model_params"]["source_path"] = os.path.join(cfg["model_params"]["source_path"], args.scene_name)
        except:
            print("OverrideError: Cannot override 'exp_name' and 'source_path' in 'model_params'. Exiting.")
            sys.exit(1)
            
    lp, op, pp = parse_cfg(cfg)
    lp.model_path = args.model_path

    print("Rendering " + args.model_path)
    
    # Generate timestamp for mesh folder
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    mesh_dir = os.path.join(args.model_path, 'train', f"mesh_{timestamp}")
    os.makedirs(mesh_dir, exist_ok=True)
    
    modules = __import__('scene')
    model_config = lp.model_config
    iteration = args.iteration
    gaussians = getattr(modules, model_config['name'])(**model_config['kwargs'])
    scene = Scene(lp, gaussians, load_iteration=iteration, shuffle=False)

    # Import gaussian_renderer module
    gaussian_renderer_modules = __import__('gaussian_renderer')

    # Handle query_label_id: -1 for all objects, -2 for all individual objects
    if args.query_label_id == -1:
        queried_object_mask = torch.ones_like(gaussians.label_ids.squeeze(), dtype=torch.bool)
        print(f"Querying all objects (query_label_id: -1)")
        print(f"Available label_ids: {torch.unique(gaussians.label_ids.squeeze()).tolist()}")
        print(f"Total number of gaussians: {queried_object_mask.sum().item()}")
        
        # Process all objects together and exit
        gaussExtractor = GaussianExtractor(gaussians, getattr(gaussian_renderer_modules, 'render'), pp, scene.background, queried_object_mask) 

        # set the active_sh to 0 to export only diffuse texture
        if gaussExtractor.gaussians.active_sh_degree != None: 
            gaussExtractor.gaussians.active_sh_degree = 0
        gaussExtractor.reconstruction(scene.getTrainCameras())
        # extract the mesh and save
        if args.unbounded:
            name = f'fuse_unbounded_all.ply'
            mesh = gaussExtractor.extract_mesh_unbounded(resolution=args.mesh_res)
        else:
            name = f'fuse_all.ply'
            depth_trunc = (gaussExtractor.radius * 2.0)*5 if args.depth_trunc < 0  else args.depth_trunc
            voxel_size = (depth_trunc / args.mesh_res) if args.voxel_size < 0 else args.voxel_size
            sdf_trunc = 5.0 * voxel_size if args.sdf_trunc < 0 else args.sdf_trunc
            mesh = gaussExtractor.extract_mesh_bounded(voxel_size=voxel_size, sdf_trunc=sdf_trunc, depth_trunc=depth_trunc)

        o3d.io.write_triangle_mesh(os.path.join(mesh_dir, name), mesh)
        print("mesh saved at {}".format(os.path.join(mesh_dir, name)))
        # post-process the mesh and save, saving the largest N clusters
        mesh_post = post_process_mesh(mesh, cluster_to_keep=args.num_cluster)
        post_name = name.replace('.ply', '_post.ply')
        o3d.io.write_triangle_mesh(os.path.join(mesh_dir, post_name), mesh_post)
        print("mesh post processed saved at {}".format(os.path.join(mesh_dir, post_name)))
        sys.exit(0)  # Exit after processing all objects together
    elif args.query_label_id == -2:
        # Export all individual objects
        available_labels = torch.unique(gaussians.label_ids.squeeze()).tolist()
        print(f"Exporting all individual objects (query_label_id: -2)")
        print(f"Available label_ids: {available_labels}")
        
        for label_id in available_labels:
            print(f"\n{'='*50}")
            print(f"Processing label_id: {label_id}")
            print(f"{'='*50}")
            
            # Create mask for current label
            current_mask = gaussians.label_ids.squeeze() == label_id
            print(f"Number of gaussians with label {label_id}: {current_mask.sum().item()}")
            
            if current_mask.sum().item() == 0:
                print(f"Warning: No gaussians found with label_id {label_id}, skipping...")
                continue
            
            # Create GaussianExtractor for current label
            current_gaussExtractor = GaussianExtractor(gaussians, getattr(gaussian_renderer_modules, 'render'), pp, scene.background, current_mask)
            
            # Set the active_sh to 0 to export only diffuse texture
            if current_gaussExtractor.gaussians.active_sh_degree != None: 
                current_gaussExtractor.gaussians.active_sh_degree = 0
            
            current_gaussExtractor.reconstruction(scene.getTrainCameras())
            
            # Extract the mesh and save
            if args.unbounded:
                name = f'fuse_unbounded_{label_id}.ply'
                mesh = current_gaussExtractor.extract_mesh_unbounded(resolution=args.mesh_res)
            else:
                name = f'fuse_{label_id}.ply'
                depth_trunc = (current_gaussExtractor.radius * 2.0)*5 if args.depth_trunc < 0 else args.depth_trunc
                voxel_size = (depth_trunc / args.mesh_res) if args.voxel_size < 0 else args.voxel_size
                sdf_trunc = 5.0 * voxel_size if args.sdf_trunc < 0 else args.sdf_trunc
                mesh = current_gaussExtractor.extract_mesh_bounded(voxel_size=voxel_size, sdf_trunc=sdf_trunc, depth_trunc=depth_trunc)

            o3d.io.write_triangle_mesh(os.path.join(mesh_dir, name), mesh)
            print(f"Mesh saved at {os.path.join(mesh_dir, name)}")
            
            # Post-process the mesh and save
            if len(mesh.vertices) > 0:
                mesh_post = post_process_mesh(mesh, cluster_to_keep=args.num_cluster)
                post_name = name.replace('.ply', '_post.ply')
                o3d.io.write_triangle_mesh(os.path.join(mesh_dir, post_name), mesh_post)
                print(f"Post-processed mesh saved at {os.path.join(mesh_dir, post_name)}")
            else:
                print(f"Warning: Mesh for label_id {label_id} has no vertices, skipping post-processing")
        
        print(f"\n{'='*50}")
        print("All individual objects exported successfully!")
        print(f"{'='*50}")
        sys.exit(0)  # Exit after processing all objects
    else:
        queried_object_mask = gaussians.label_ids.squeeze() == args.query_label_id
        
        # Debug: Check if the queried object exists
        print(f"Querying object with label_id: {args.query_label_id}")
        print(f"Available label_ids: {torch.unique(gaussians.label_ids.squeeze()).tolist()}")
        print(f"Number of gaussians with queried label: {queried_object_mask.sum().item()}")
        
        if queried_object_mask.sum().item() == 0:
            print(f"Warning: No gaussians found with label_id {args.query_label_id}")
            print("Available label_ids:", torch.unique(gaussians.label_ids.squeeze()).tolist())
            sys.exit(1)

    gaussExtractor = GaussianExtractor(gaussians, getattr(gaussian_renderer_modules, 'render'), pp, scene.background, queried_object_mask) 

    # set the active_sh to 0 to export only diffuse texture
    if gaussExtractor.gaussians.active_sh_degree != None: 
        gaussExtractor.gaussians.active_sh_degree = 0
    gaussExtractor.reconstruction(scene.getTrainCameras())
    # extract the mesh and save
    if args.unbounded:
        name = f'fuse_unbounded_{args.query_label_id}.ply'
        mesh = gaussExtractor.extract_mesh_unbounded(resolution=args.mesh_res)
    else:
        name = f'fuse_{args.query_label_id}.ply'
        depth_trunc = (gaussExtractor.radius * 2.0)*5 if args.depth_trunc < 0  else args.depth_trunc
        # depth_trunc = (gaussExtractor.radius * 2.0) if args.depth_trunc < 0  else args.depth_trunc
        voxel_size = (depth_trunc / args.mesh_res) if args.voxel_size < 0 else args.voxel_size
        sdf_trunc = 5.0 * voxel_size if args.sdf_trunc < 0 else args.sdf_trunc
        mesh = gaussExtractor.extract_mesh_bounded(voxel_size=voxel_size, sdf_trunc=sdf_trunc, depth_trunc=depth_trunc)

    o3d.io.write_triangle_mesh(os.path.join(mesh_dir, name), mesh)
    print("mesh saved at {}".format(os.path.join(mesh_dir, name)))
    # post-process the mesh and save, saving the largest N clusters
    mesh_post = post_process_mesh(mesh, cluster_to_keep=args.num_cluster)
    post_name = name.replace('.ply', '_post.ply')
    o3d.io.write_triangle_mesh(os.path.join(mesh_dir, post_name), mesh_post)
    print("mesh post processed saved at {}".format(os.path.join(mesh_dir, post_name)))