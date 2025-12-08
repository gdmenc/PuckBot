import os
from textwrap import dedent

def file_uri(path: str) -> str:
    abs_path = path if os.path.isabs(path) else os.path.abspath(path)
    return f"file://{abs_path}"

# Paths
repo_dir = os.path.dirname(os.path.dirname(os.path.dirname(__file__)))
assets_dir = os.path.join(repo_dir, "scripts", "env", "assets", "models")

ground_uri = file_uri(os.path.join(assets_dir, "ground", "ground.sdf"))
table_uri = file_uri(os.path.join(assets_dir, "air_hockey_table", "table.xml"))
paddle_uri = file_uri(os.path.join(assets_dir, "paddle", "paddle.sdf"))
wsg_uri = file_uri(os.path.join(assets_dir, "schunk_wsg_50", "schunk_wsg_50_with_tip.sdf"))
puck_uri = file_uri(os.path.join(assets_dir, "puck", "puck.sdf"))

print("GROUND URI:", ground_uri)
print("Exists:", os.path.exists(os.path.join(assets_dir, "ground", "ground.sdf")))
print("TABLE URI:", table_uri)
print("Exists:", os.path.exists(os.path.join(assets_dir, "air_hockey_table", "table.xml")))
print("PADDLE URI:", paddle_uri)
print("Exists:", os.path.exists(os.path.join(assets_dir, "paddle", "paddle.sdf")))
print("WSG URI:", wsg_uri)
print("Exists:", os.path.exists(os.path.join(assets_dir, "schunk_wsg_50", "schunk_wsg_50_with_tip.sdf")))
print("PUCK URI:", puck_uri)
print("Exists:", os.path.exists(os.path.join(assets_dir, "puck", "puck.sdf")))

yaml = """
directives:

  ## GROUND ## 
  - add_model:
      name: ground
      file: FILE_GROUND
  - add_weld:
      parent: world
      child: ground::ground_link
      X_PC:
        translation: [0.0, 0.0, -0.05]
        rotation: !Rpy { deg: [0, 0, -90] }

  ### TABLE ###
  - add_model:
      name: table
      file: FILE_TABLE

  ### LEFT IIWA ###
  - add_model:
      name: left_iiwa
      file: package://drake_models/iiwa_description/sdf/iiwa7_with_box_collision.sdf
  - add_weld:
      parent: world
      child: left_iiwa::iiwa_link_0
      X_PC:
        translation: [-1.3, 0.0, 0.0]
        rotation: !Rpy { deg: [0, 0, 180] }

  - add_model:
      name: left_wsg
      file: FILE_WSG
  - add_weld:
      parent: left_iiwa::iiwa_link_7
      child: left_wsg::body
      X_PC:
        translation: [0.0, 0.0, 0.09]
        rotation: !Rpy { deg: [90, 0, 90] }

  ### RIGHT IIWA ###
  - add_model:
      name: right_iiwa
      file: package://drake_models/iiwa_description/sdf/iiwa7_with_box_collision.sdf
  - add_weld:
      parent: world
      child: right_iiwa::iiwa_link_0
      X_PC:
        translation: [1.3, 0.0, 0.0]
        rotation: !Rpy { deg: [0, 0, 0] }

  - add_model:
      name: right_wsg
      file: FILE_WSG
  - add_weld:
      parent: right_iiwa::iiwa_link_7
      child: right_wsg::body
      X_PC:
        translation: [0.0, 0.0, 0.09]
        rotation: !Rpy { deg: [90, 0, 90] }

  ### PUCK ###
  - add_model:
      name: puck
      file: FILE_PUCK
  - add_weld:
      parent: world
      child: puck::puck_body_link
      X_PC:
        translation: [0.0, 0.0, 0.04]
        rotation: !Rpy { deg: [0, 0, 0] }

  ### FREE PADDLES ###
  - add_model:
      name: left_paddle
      file: FILE_PADDLE
  - add_weld:
      parent: world
      child: left_paddle::paddle_body_link
      X_PC:
        translation: [0.0, 0.0, 0.04]
        rotation: !Rpy { deg: [0, 0, 0] }

"""

  # - add_model:
  #     name: right_paddle
  #     file: FILE_PADDLE
  #     default_free_body_pose:
  #       paddle_body_link:
  #         translation: [-0.10, 0.0, 0.50]
  #         rotation: !Rpy { deg: [0, 0, 0] }


yaml = yaml.replace("FILE_TABLE", table_uri)
yaml = yaml.replace("FILE_PADDLE", paddle_uri)
yaml = yaml.replace("FILE_WSG", wsg_uri)
yaml = yaml.replace("FILE_GROUND", ground_uri)
yaml = yaml.replace("FILE_PUCK", puck_uri)

out_path = os.path.join(repo_dir, "scripts", "env", "scenario/puckbot_scene.yaml")
os.makedirs(os.path.dirname(out_path), exist_ok=True)
with open(out_path, "w") as f:
    f.write(dedent(yaml).lstrip())

print("Wrote directives →", out_path)
