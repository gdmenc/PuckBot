import os
from textwrap import dedent

def file_uri(path: str) -> str:
    abs_path = path if os.path.isabs(path) else os.path.abspath(path)
    return f"file://{abs_path}"

# Paths
def get_asset_paths():
    repo_dir = os.path.dirname(os.path.dirname(os.path.dirname(__file__)))
    assets_dir = os.path.join(repo_dir, "scripts", "env", "assets", "models")
    return {
        "ground": file_uri(os.path.join(assets_dir, "ground", "ground.sdf")),
        "table": file_uri(os.path.join(assets_dir, "air_hockey_table", "table.sdf")),
        "paddle": file_uri(os.path.join(assets_dir, "paddle", "paddle.sdf")),
        "wsg": file_uri(os.path.join(assets_dir, "schunk_wsg_50", "schunk_wsg_50_with_tip.sdf")),
        "puck": file_uri(os.path.join(assets_dir, "puck", "puck.sdf")),
    }

def generate_scenario_yaml(num_arms: int = 2, save_file: bool = True, weld_paddles: bool = False) -> str:
    paths = get_asset_paths()
    
    # Common directives (Ground, Table, Puck)
    # Table weld adjusted to 0.1 to match user spec (surface at 0.1)
    # The table link origin is the top surface (based on visual offset -0.0505 and height 0.101).
    yaml_base = f"""
directives:

  ## GROUND ## 
  - add_model:
      name: ground
      file: {paths['ground']}
  - add_weld:
      parent: world
      child: ground::ground_link
      X_PC:
        translation: [0.0, 0.0, -0.05]
        rotation: !Rpy {{ deg: [0, 0, -90] }}

  ### TABLE ###
  - add_model:
      name: air_hockey_table
      file: {paths['table']}
  - add_weld:
      parent: world
      child: air_hockey_table::table_body
      X_PC:
        translation: [0.0, 0.0, 0.10]
        rotation: !Rpy {{ deg: [0, 0, 0] }}

  ### PUCK ###
  - add_model:
      name: puck
      file: {paths['puck']}
"""

    yaml_arms = ""
    
    # Right Arm (Player)
    yaml_arms += f"""
  ### RIGHT IIWA ###
  - add_model:
      name: right_iiwa
      file: package://drake_models/iiwa_description/sdf/iiwa7_with_box_collision.sdf
  - add_weld:
      parent: world
      child: right_iiwa::iiwa_link_0
      X_PC:
        translation: [1.3, 0.0, 0.0]
        rotation: !Rpy {{ deg: [0, 0, 0] }}

  - add_model:
      name: right_wsg
      file: {paths['wsg']}
  - add_weld:
      parent: right_iiwa::iiwa_link_7
      child: right_wsg::body
      X_PC:
        translation: [0.0, 0.0, 0.09]
        rotation: !Rpy {{ deg: [90, 0, 90] }}
"""

    if num_arms == 2:
        yaml_arms += f"""
  ### LEFT IIWA ###
  - add_model:
      name: left_iiwa
      file: package://drake_models/iiwa_description/sdf/iiwa7_with_box_collision.sdf
  - add_weld:
      parent: world
      child: left_iiwa::iiwa_link_0
      X_PC:
        translation: [-1.3, 0.0, 0.0]
        rotation: !Rpy {{ deg: [0, 0, 180] }}

  - add_model:
      name: left_wsg
      file: {paths['wsg']}
  - add_weld:
      parent: left_iiwa::iiwa_link_7
      child: left_wsg::body
      X_PC:
        translation: [0.0, 0.0, 0.09]
        rotation: !Rpy {{ deg: [90, 0, 90] }}
"""

    # Paddles 
    yaml_paddles = ""
    # Right Paddle
    if weld_paddles:
        yaml_paddles += f"""
  - add_model:
      name: right_paddle
      file: {paths['paddle']}
  - add_weld:
      parent: right_wsg::body
      child: right_paddle::paddle_body_link
      X_PC:
        translation: [0.0, 0.08, -0.07]
        rotation: !Rpy {{ deg: [0, 0, 0] }}
"""
    else:
        yaml_paddles += f"""
  - add_model:
      name: right_paddle
      file: {paths['paddle']}
  # Paddle is free-floating (no weld)
"""

    if num_arms == 2:
        # Left Paddle
        if weld_paddles:
            yaml_paddles += f"""
  - add_model:
      name: left_paddle
      file: {paths['paddle']}
  - add_weld:
      parent: left_wsg::body
      child: left_paddle::paddle_body_link
      X_PC:
        translation: [0.0, 0.08, -0.07]
        rotation: !Rpy {{ deg: [0, 0, 0] }}
"""
        else:
            yaml_paddles += f"""
  - add_model:
      name: left_paddle
      file: {paths['paddle']}
  # Paddle is free-floating (no weld)
"""

    final_yaml = dedent(yaml_base + yaml_arms + yaml_paddles)
    
    if save_file:
        repo_dir = os.path.dirname(os.path.dirname(os.path.dirname(__file__)))
        scenario_dir = os.path.join(repo_dir, "scripts", "env", "scenario")
        os.makedirs(scenario_dir, exist_ok=True)
        
        filename = "single_arm.yaml" if num_arms == 1 else "two_arm.yaml"
        file_path = os.path.join(scenario_dir, filename)
        
        with open(file_path, "w") as f:
            f.write(final_yaml)
        print(f"[INFO] Saved scenario to: {file_path}")
        
    return final_yaml
