#
from tensorboard.backend.event_processing import event_accumulator
from tensorboard.summary.writer.event_file_writer import EventFileWriter
from tensorboard.compat.proto import event_pb2
from tensorboard.compat.proto import summary_pb2
import os

# 输入/输出文件路径
dir_path = r'H:\My-JDD\logs\exp001_netdepcv_L1_fft_vgg\lightning_logs\version_0'

# 自动匹配以 .0 结尾的文件
files = os.listdir(dir_path)
tf_log_files = [f for f in files if f.endswith('.0')]
if not tf_log_files:
    raise FileNotFoundError(f"未在 {dir_path} 目录下找到以 .0 结尾的文件")
# 若有多个符合条件的文件，取第一个
tf_log_file = tf_log_files[0]

input_file = os.path.join(dir_path, tf_log_file)
output_file = dir_path

# 读取事件文件
ea = event_accumulator.EventAccumulator(input_file, size_guidance={
    event_accumulator.SCALARS: 0,
    event_accumulator.IMAGES: 0,
    event_accumulator.HISTOGRAMS: 0,
    event_accumulator.COMPRESSED_HISTOGRAMS: 0,
    event_accumulator.AUDIO: 0
})
ea.Reload()

# 手动创建 EventFileWriter 实例
writer = EventFileWriter(output_file)
try:
    # 1. 处理标量数据
    for tag in ea.Tags().get("scalars", []):
        scalar_events = ea.Scalars(tag)
        if "val_clip" in tag or "val_frame" in tag:
            num_events = len(scalar_events)
            if num_events == 0:
                continue
            elif num_events == 1:
                # 只有一条记录时保留该记录
                selected_events = [scalar_events[0]]
            else:
                # 计算中间记录的索引
                mid_idx = num_events // 2
                # 保留第一条、中间一条和最后一条记录
                selected_events = [scalar_events[0], scalar_events[mid_idx], scalar_events[-1]]
        else:
            # 其他标量数据保留所有记录
            selected_events = scalar_events

        for event in selected_events:
            proto_event = event_pb2.Event(
                wall_time=event.wall_time,
                step=event.step,
                summary=summary_pb2.Summary(
                    value=[
                        summary_pb2.Summary.Value(
                            tag=tag,
                            simple_value=event.value
                        )
                    ]
                )
            )
            writer.add_event(proto_event)

    # 2. 处理图像数据：保存 5 个时间点的记录
    for tag in ea.Tags().get("images", []):
        image_events = ea.Images(tag)
        num_events = len(image_events)

        if num_events == 0:
            continue
        elif num_events == 1:
            # 只有一条记录时保留该记录
            selected_image_events = [image_events[0]]
        else:
            # 计算 5 个时间点的索引
            ratios = [0, 0.2, 0.4, 0.6, 0.8, 1]
            selected_image_events = []
            for ratio in ratios:
                idx = min(int(ratio * (num_events - 1)), num_events - 1)
                selected_image_events.append(image_events[idx])

        for image_event in selected_image_events:
            event = event_pb2.Event(
                wall_time=image_event.wall_time,
                step=image_event.step,
                summary=summary_pb2.Summary(
                    value=[
                        summary_pb2.Summary.Value(
                            tag=tag,
                            image=summary_pb2.Summary.Image(
                                encoded_image_string=image_event.encoded_image_string,
                                height=image_event.height,
                                width=image_event.width
                            )
                        )
                    ]
                )
            )
            writer.add_event(event)

    # 3. 处理直方图数据
    for tag in ea.Tags().get("histograms", []):
        for hist_event in ea.Histograms(tag):
            event = event_pb2.Event(
                wall_time=hist_event.wall_time,
                step=hist_event.step,
                summary=summary_pb2.Summary(
                    value=[
                        summary_pb2.Summary.Value(
                            tag=tag,
                            histo=summary_pb2.HistogramProto(
                                min=hist_event.min,
                                max=hist_event.max,
                                num=hist_event.num,
                                sum=hist_event.sum,
                                sum_squares=hist_event.sum_squares,
                                bucket_limit=hist_event.bucket_limit,
                                bucket=hist_event.bucket
                            )
                        )
                    ]
                )
            )
            writer.add_event(event)

    # 4. 处理 HPARAM 数据，完整保留
    for event in ea._generator.Load():
        if event.HasField('session_log') and event.session_log.status == event_pb2.SessionLog.START:
            writer.add_event(event)

    # 5. 处理图数据
    if "graph" in ea.Tags():
        try:
            graph_def = ea.Graph()
            event = event_pb2.Event(graph_def=graph_def.SerializeToString())
            writer.add_event(event)
        except ValueError:
            pass

except Exception as e:
    print(f"处理过程中出现错误: {e}")
finally:
    # 手动关闭 writer
    try:
        writer.close()
    except Exception as e:
        print(f"关闭 writer 时出现错误: {e}")