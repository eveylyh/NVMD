import os
import pandas as pd
import shutil

# 文件路径配置
csv_path = "E:\\PycharmProjects\\CLIP_prefix_caption-main\\dataprocess\\mediaeval2015\\tweets.csv"
images_source_base = "E:\\PycharmProjects\\CLIP_prefix_caption-main\\dataprocess\\mediaeval2015\\Medieval2015_DevSet_Images\\"
images_dest_base = "E:\\PycharmProjects\\pmccRes\\total_dataset\\twitter\\"

# 事件名称映射
event_name_map = {
    "sandyA": "Hurricane Sandy", "sandyB": "Hurricane Sandy",
    "boston": "BostonMarathon", "columbianChemicals": "Columbian Chemicals",
    "bringback": "Bring Back Our Girls", "passport": "Passport",
    "underwater": "Underwater bedroom", "elephant": "Rock Elephant",
    "sochi": "Sochi Olympics", "malaysia": "Malaysia Airlines",
    "livr": "Livr", "pigFish": "Pig Fish"
}

# 读取CSV文件，使用制表符作为分隔符
df = pd.read_csv(csv_path, sep='\t')

# 初始化计数器
copied_files_count = 0

# 遍历数据
for index, row in df.iterrows():
    tweet_id = row['tweetId']
    image_id = row['imageId']
    parts = image_id.split('_')

    # 确定事件名和标签
    event_name = event_name_map.get(parts[0], parts[0])
    label = parts[1] if len(parts) > 2 else "fake"

    # 构建源和目标路径
    if event_name == "Malaysia Airlines":
        source_path = os.path.join(images_source_base, event_name, image_id + ".jpg")
    else:
        source_path = os.path.join(images_source_base, event_name, label + 's', image_id + ".jpg")
    dest_path = os.path.join(images_dest_base, label, str(tweet_id))

    # 检查源文件是否存在
    if not os.path.exists(source_path):
        continue

    # 检查目标文件夹是否存在，如果不存在则跳过
    if not os.path.exists(dest_path):
        continue

    # 执行复制操作
    shutil.copy(source_path, dest_path)
    copied_files_count += 1

# 输出结果
print(f"成功复制的图片数量：{copied_files_count}")
