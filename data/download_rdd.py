from roboflow import Roboflow
import os

# 1. Initialize with your key
rf = Roboflow(api_key="94ux6QIXWE5JJILBCx6L")

# 2. Access the RDD2022 India project
# Workspace: prakhar-kpb1v, Project: rdd2022-india-il8ju
project = rf.workspace("prakhar-kpb1v").project("rdd2022-india-il8ju")
version = project.version(1)

# 3. Download in YOLOv11 format
# This will create a folder named 'RDD2022-India-1' in your current directory
dataset = version.download("yolov11")

print("Download Complete! Your data is in:", os.path.abspath(dataset.location))