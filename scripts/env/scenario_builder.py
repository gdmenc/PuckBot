import os
from textwrap import dedent

def file_uri(path: str) -> str:
    abs_path = path if os.path.isabs(path) else os.path.abspath(path)
    return f"file://{abs_path}"

repo_dir = os.path.dirname(os.path.dirname(os.path.dirname(__file__)))
assets_dir = os.path.join(repo_dir, "scripts", "env", "assets", "models")

paddle_uri = file_uri(os.path.join(assets_dir, "paddle.sdf"))
table_uri = file_uri(os.path.join(assets_dir, "table.xml"))

yaml = """
directives:

  ### TABLE ###
  - add_model:
      name: table
      file: FILE_TABLE

  ### LEFT IIWA ###
  - add_model:
      name: left_iiwa
      file: package://drake_models/iiwa_description/sdf/iiwa7_no_collision.sdf
  - add_weld:
      parent: world
      child: left_iiwa::iiwa_link_0
      X_PC:
        translation: [0.0, 0.85, 0.0]
        rotation: !Rpy { deg: [0, 0, 180] }

  - add_model:
      name: left_wsg
      file: package://manipulation/hydro/schunk_wsg_50_with_tip.sdf
  - add_weld:
      parent: left_iiwa::iiwa_link_7
      child: left_wsg::body
      X_PC:
        translation: [0.0, 0.0, 0.09]
        rotation: !Rpy { deg: [90, 0, 90] }

  ### RIGHT IIWA ###
  - add_model:
      name: right_iiwa
      file: package://drake_models/iiwa_description/sdf/iiwa7_no_collision.sdf
  - add_weld:
      parent: world
      child: right_iiwa::iiwa_link_0
      X_PC:
        translation: [0.0, -0.85, 0.0]
        rotation: !Rpy { deg: [0, 0, 0] }

  - add_model:
      name: right_wsg
      file: package://manipulation/hydro/schunk_wsg_50_with_tip.sdf
  - add_weld:
      parent: right_iiwa::iiwa_link_7
      child: right_wsg::body
      X_PC:
        translation: [0.0, 0.0, 0.09]
        rotation: !Rpy { deg: [90, 0, 90] }

  ### FREE PADDLES ###
  - add_model:
      name: left_paddle
      file: FILE_PADDLE
      default_free_body_pose:
        paddle_body_link:
          translation: [0.0, 0.45, 0.10]
          rotation: !Rpy { deg: [0, 0, 180] }

  - add_model:
      name: right_paddle
      file: FILE_PADDLE
      default_free_body_pose:
        paddle_body_link:
          translation: [0.0, -0.45, 0.10]
          rotation: !Rpy { deg: [0, 0, 0] }
"""

yaml = yaml.replace("FILE_TABLE", table_uri)
yaml = yaml.replace("FILE_PADDLE", paddle_uri)

out_path = os.path.join(repo_dir, "scripts", "env", "puckbot_scene.yaml")
os.makedirs(os.path.dirname(out_path), exist_ok=True)
with open(out_path, "w") as f:
    f.write(dedent(yaml).lstrip())

print("Wrote directives to:", out_path)
print("Table URI:", table_uri)
print("Paddle URI:", paddle_uri)
