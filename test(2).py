import cv2
import mediapipe as mp
import numpy as np
import time
import os
import threading
import pyttsx3
import pygame
from math import acos, degrees, sqrt
from queue import Queue

# ==================== Configuration ====================
# Image paths
BASE_DIR = r"C:\Users\Lucien\Desktop\third"

# Target image scale
TARGET_IMG_SCALE = 0.25

# Define multiple poses (each with reference, target and hint images)
POSE_SEQUENCE = [
    {
        "name": "Side Plank",
        "ref_img": os.path.join(BASE_DIR, "picture1.jpg"),
        "target_img": os.path.join(BASE_DIR, "picture2.png"),
        "hint_img": os.path.join(BASE_DIR, "picture3.png"),
        "success_sound": "well_done.wav",  # 添加成功音效文件
        "success_speech": "Well done!"  # 添加成功语音文本
    },
    {
        "name": "Tree Pose",
        "ref_img": os.path.join(BASE_DIR, "pose2_ref.png"),
        "target_img": os.path.join(BASE_DIR, "pose2_target.png"),
        "hint_img": os.path.join(BASE_DIR, "picture3.png"),
        "success_sound": "well_done.wav",
        "success_speech": "Please go up!"
    },
    {
        "name": "Downward Dog",
        "ref_img": os.path.join(BASE_DIR, "pose3_ref.png"),
        "target_img": os.path.join(BASE_DIR, "pose3_target.png"),
        "hint_img": os.path.join(BASE_DIR, "picture3.png"),
        "success_sound": "congratulations.wav",
        "success_speech": "All poses completed! Congratulations!"
    }
]

# Game parameters
REQUIRED_DURATION = 5.0  # 改为5秒 Hold pose for 5 seconds
MAX_ANGLE_DEVIATION = 40.0  # Increased deviation threshold
MIN_VISIBILITY = 0.5  # Lower visibility threshold
PINK = (203, 192, 255)  # BGR color for pink arrows
ARROW_SCALE = 0.3  # Arrow scaling factor
THUMBS_UP_DISPLAY_DURATION = 2.0  # 点赞图片显示时间（秒）

# ==================== Initialization ====================
mp_drawing = mp.solutions.drawing_utils
mp_pose = mp.solutions.pose

# 初始化pygame用于播放音效
pygame.mixer.init()

# Speech queue and thread control
speech_queue = Queue()
speech_thread_running = True

# Initialize speech engine
engine = pyttsx3.init()
voices = engine.getProperty('voices')
for voice in voices:
    if "english" in voice.id.lower() and "female" in voice.name.lower():
        engine.setProperty('voice', voice.id)
        break
    elif "english" in voice.id.lower() and "female" in voice.gender.lower():
        engine.setProperty('voice', voice.id)
        break
engine.setProperty('rate', 150)
engine.setProperty('volume', 0.8)


# ==================== Helper Functions ====================
def load_reference_pose(image_path):
    """Load reference pose image"""
    image = cv2.imread(image_path)
    if image is None:
        raise ValueError(f"Failed to load image: {image_path}")
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    with mp_pose.Pose(static_image_mode=True, min_detection_confidence=0.7) as pose:
        results = pose.process(image_rgb)
        if not results.pose_landmarks:
            raise ValueError("No pose detected in reference image")
        return results.pose_landmarks


def blend_transparent(background, overlay, x, y):
    """Blend transparent image onto background"""
    if overlay is None:
        return background

    if overlay.shape[2] == 4:  # With alpha channel
        overlay_img = overlay[:, :, :3]
        alpha = overlay[:, :, 3:] / 255.0
        h, w = overlay_img.shape[:2]
        roi = background[y:y + h, x:x + w]
        background[y:y + h, x:x + w] = (1.0 - alpha) * roi + alpha * overlay_img
    else:  # Without alpha channel
        background[y:y + overlay.shape[0], x:x + overlay.shape[1]] = overlay
    return background


def draw_text_with_border(img, text, position, font_scale=1.0, thickness=2, color=(255, 255, 255)):
    """Draw text with black border"""
    x, y = position
    # Black border
    cv2.putText(img, text, (x, y), cv2.FONT_HERSHEY_SIMPLEX, font_scale, (0, 0, 0), thickness + 2, cv2.LINE_AA)
    # Colored text
    cv2.putText(img, text, (x, y), cv2.FONT_HERSHEY_SIMPLEX, font_scale, color, thickness, cv2.LINE_AA)


def calculate_limb_deviation(user_pose, ref_pose, i, j, k):
    """Calculate limb deviation angle"""
    # Get reference vectors
    ref_vec1 = np.array([ref_pose.landmark[j].x - ref_pose.landmark[i].x,
                         ref_pose.landmark[j].y - ref_pose.landmark[i].y])
    ref_vec2 = np.array([ref_pose.landmark[k].x - ref_pose.landmark[j].x,
                         ref_pose.landmark[k].y - ref_pose.landmark[j].y])

    # Get user vectors
    user_vec1 = np.array([user_pose.landmark[j].x - user_pose.landmark[i].x,
                          user_pose.landmark[j].y - user_pose.landmark[i].y])
    user_vec2 = np.array([user_pose.landmark[k].x - user_pose.landmark[j].x,
                          user_pose.landmark[k].y - user_pose.landmark[j].y])

    # Calculate vector angles
    def vector_angle(v1, v2):
        cos_theta = np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2))
        return degrees(acos(np.clip(cos_theta, -1.0, 1.0)))

    ref_angle = vector_angle(ref_vec1, ref_vec2)
    user_angle = vector_angle(user_vec1, user_vec2)

    return abs(ref_angle - user_angle)


def calculate_vector_direction(ref_pose, user_pose, i, j):
    """Calculate direction difference between reference and user vectors"""
    # Reference vector (from point i to j)
    ref_vector = np.array([
        ref_pose.landmark[j].x - ref_pose.landmark[i].x,
        ref_pose.landmark[j].y - ref_pose.landmark[i].y
    ])

    # User vector (from point i to j)
    user_vector = np.array([
        user_pose.landmark[j].x - user_pose.landmark[i].x,
        user_pose.landmark[j].y - user_pose.landmark[i].y
    ])

    # Calculate direction difference
    direction_difference = ref_vector - user_vector

    # Normalize vectors
    norm_ref = np.linalg.norm(ref_vector)
    norm_user = np.linalg.norm(user_vector)

    if norm_ref > 0 and norm_user > 0:
        ref_vector = ref_vector / norm_ref
        user_vector = user_vector / norm_user

        # Calculate angle difference (in degrees)
        cos_theta = np.dot(ref_vector, user_vector)
        angle_difference = degrees(acos(np.clip(cos_theta, -1.0, 1.0)))

        # Calculate cross product to determine direction
        cross_product = np.cross(ref_vector, user_vector)
        if cross_product < 0:
            angle_difference = -angle_difference

        return direction_difference, angle_difference

    return np.array([0, 0]), 0


def speech_worker():
    """Speech playback worker thread"""
    global speech_thread_running
    while speech_thread_running or not speech_queue.empty():
        if not speech_queue.empty():
            text = speech_queue.get()
            try:
                engine.say(text)
                engine.runAndWait()
            except Exception as e:
                print(f"Speech playback error: {e}")
        time.sleep(0.1)


def play_encouragement():
    """Non-blocking encouragement speech"""
    speech_queue.put("Keep going, perseverance is victory!")


def play_sound_effect(sound_file):
    """播放音效"""
    try:
        sound_path = os.path.join(BASE_DIR, sound_file)
        if os.path.exists(sound_path):
            pygame.mixer.music.load(sound_path)
            pygame.mixer.music.play()
            return True
        else:
            print(f"Sound file not found: {sound_path}")
            return False
    except Exception as e:
        print(f"Error playing sound: {e}")
        return False


# ==================== Main Function ====================
def main():
    global speech_thread_running

    # 加载点赞图片
    thumbs_up_image = cv2.imread(os.path.join(BASE_DIR, "thumbs up.png"), cv2.IMREAD_UNCHANGED)
    if thumbs_up_image is None:
        print("无法加载点赞图片")
        thumbs_up_image = None
    else:
        # 调整点赞图片大小（宽度为屏幕宽度的20%）
        thumbs_up_width = int(640 * 0.4)
        thumbs_up_image = cv2.resize(
            thumbs_up_image,
            (thumbs_up_width, int(thumbs_up_width * thumbs_up_image.shape[0] / thumbs_up_image.shape[1]))
        )

    # Start speech thread
    speech_thread = threading.Thread(target=speech_worker, daemon=True)
    speech_thread.start()

    # Setup camera
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Failed to open camera")
        speech_thread_running = False
        return

    # 获取屏幕分辨率
    screen_width = 1920  # 默认值
    screen_height = 1080  # 默认值
    try:
        # 尝试获取实际屏幕分辨率
        screen_width = cap.get(cv2.CAP_PROP_FRAME_WIDTH)
        screen_height = cap.get(cv2.CAP_PROP_FRAME_HEIGHT)
    except:
        pass

    # 创建全屏窗口
    cv2.namedWindow('Precision Pose Challenge', cv2.WND_PROP_FULLSCREEN)
    cv2.setWindowProperty('Precision Pose Challenge', cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)

    # Initialize game state
    score = 0
    is_matching = False
    start_matching_time = 0
    feedback_msg = "Adjust your pose..."
    start_time = time.time()
    current_pose_index = 0
    pose_completed = [False] * len(POSE_SEQUENCE)
    encouragement_played = False
    last_deviation_text = ""
    final_congrats_played = False

    # 新增：点赞图片显示状态
    thumbs_up_display_start = 0
    thumbs_up_displaying = False

    # Load current pose resources
    try:
        current_pose = POSE_SEQUENCE[current_pose_index]
        ref_pose = load_reference_pose(current_pose["ref_img"])
        target_image = cv2.imread(current_pose["target_img"], cv2.IMREAD_UNCHANGED)
        hint_image = cv2.imread(current_pose["hint_img"], cv2.IMREAD_UNCHANGED)

        # Resize target image
        target_image = cv2.resize(
            target_image,
            (int(screen_width * TARGET_IMG_SCALE),
             int(screen_width * TARGET_IMG_SCALE * target_image.shape[0] / target_image.shape[1]))
        )
    except Exception as e:
        print(f"Image loading error: {e}")
        speech_thread_running = False
        return

    # Define limb connections (shoulder-elbow-wrist, hip-knee-ankle)
    limb_connections = [
        (11, 13, 15, "Right Arm"),  # Left shoulder-left elbow-left wrist
        (12, 14, 16, "Left Arm") ,# Right shoulder-right elbow-right wrist
        (23, 25, 27, "Right Leg")  # Left hip-left knee-left ankle
        # (24, 26, 28, "Left Leg")  # Right hip-right knee-right ankle
    ]

    # Main loop
    with mp_pose.Pose(min_detection_confidence=0.7, min_tracking_confidence=0.7) as pose:
        while cap.isOpened():
            success, image = cap.read()
            if not success:
                continue
            image = cv2.flip(image, 1)

            # 获取屏幕尺寸
            h, w = image.shape[:2]

            # 如果摄像头分辨率小于屏幕分辨率，则填充至全屏
            if w < screen_width or h < screen_height:
                # 创建一个黑色背景
                fullscreen_image = np.zeros((int(screen_height), int(screen_width), 3), dtype=np.uint8)

                # 计算居中位置
                x_offset = (int(screen_width) - w) // 2
                y_offset = (int(screen_height) - h) // 2

                # 将摄像头图像放置在中央
                fullscreen_image[y_offset:y_offset + h, x_offset:x_offset + w] = image
                image = fullscreen_image
                h, w = image.shape[:2]

            # Resize hint image
            hint_width = int(w * 0.13)
            hint_img_resized = cv2.resize(
                hint_image,
                (hint_width, int(hint_width * hint_image.shape[0] / hint_image.shape[1]))
            )

            # Process image
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            image.flags.writeable = False++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++6
            results = pose.process(image)
            image.flags.writeable = True
            image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

            # Display target image (top center) and calculate its bottom position
            target_x = (w - target_image.shape[1]) // 2
            image = blend_transparent(image, target_image, target_x, 30)
            target_img_bottom = 20 + target_image.shape[0]  # 目标图像底部位置

            # Initialize variables
            max_deviation_joint = None
            max_deviation_limb = None
            max_deviation = 0
            all_limbs_valid = False
            missing_limbs = []
            show_hint = False
            arrow_drawn = False  # Flag if arrow was drawn

            # Pose detection and scoring
            if results.pose_landmarks and not pose_completed[current_pose_index]:
                # 不再绘制骨骼绑定 - 注释掉这行代码
                # mp_drawing.draw_landmarks(
                #    image, results.pose_landmarks, mp_pose.POSE_CONNECTIONS,
                #    mp_drawing.DrawingSpec(color=(245, 117, 66), thickness=2, circle_radius=2),
                #    mp_drawing.DrawingSpec(color=(245, 66, 230), thickness=2, circle_radius=2))

                # Check pose match
                current_time = time.time()
                all_limbs_valid = True
                max_deviation = 0
                max_deviation_limb = None
                max_deviation_joint = None
                missing_limbs = []
                last_deviation_text = ""

                for i, j, k, name in limb_connections:
                    # Check if limb is visible in reference pose
                    ref_visible = (ref_pose.landmark[i].visibility > MIN_VISIBILITY and
                                   ref_pose.landmark[j].visibility > MIN_VISIBILITY and
                                   ref_pose.landmark[k].visibility > MIN_VISIBILITY)

                    if not ref_visible:
                        continue

                    # Check if limb is visible in user pose
                    user_visible = (results.pose_landmarks.landmark[i].visibility > MIN_VISIBILITY and
                                    results.pose_landmarks.landmark[j].visibility > MIN_VISIBILITY and
                                    results.pose_landmarks.landmark[k].visibility > MIN_VISIBILITY)

                    if not user_visible:
                        missing_limbs.append(name)
                        all_limbs_valid = False
                        continue

                    # Calculate deviation
                    deviation = calculate_limb_deviation(results.pose_landmarks, ref_pose, i, j, k)

                    # Calculate direction difference
                    direction_diff, angle_diff = calculate_vector_direction(ref_pose, results.pose_landmarks, i, j)

                    # Update max deviation info
                    if deviation > max_deviation:
                        max_deviation = deviation
                        max_deviation_limb = name
                        joint_x = int(results.pose_landmarks.landmark[j].x * w)
                        joint_y = int(results.pose_landmarks.landmark[j].y * h)
                        max_deviation_joint = (joint_x, joint_y)
                        last_deviation_text = f"{name} dev: {deviation:.1f}°"

                    # Draw arrow if deviation exceeds threshold
                    if deviation > MAX_ANGLE_DEVIATION:
                        all_limbs_valid = False
                        arrow_drawn = True

                        # Calculate arrow start point (joint j)
                        start_point = (joint_x, joint_y)

                        # Calculate arrow direction (scaled to screen)
                        direction_diff *= w * ARROW_SCALE

                        # Calculate arrow end point
                        end_point = (
                            int(start_point[0] + direction_diff[0]),
                            int(start_point[1] + direction_diff[1])
                        )

                        # Draw arrow
                        cv2.arrowedLine(image, start_point, end_point, PINK, 3, tipLength=0.3)

                        # Draw limb name near arrow
                        draw_text_with_border(image, name, (start_point[0] + 10, start_point[1]), 0.7, 1)

                        # Draw deviation value near arrow
                        draw_text_with_border(image, f"{deviation:.1f}°",
                                              (start_point[0] + 10, start_point[1] + 20),
                                              0.6, 1, (50, 200, 255))

                # Update matching state
                if all_limbs_valid and max_deviation <= MAX_ANGLE_DEVIATION:
                    if not is_matching:
                        is_matching = True
                        start_matching_time = current_time
                        feedback_msg = "Hold pose..."
                        encouragement_played = False
                    else:
                        duration = current_time - start_matching_time
                        if duration >= REQUIRED_DURATION:
                            pose_completed[current_pose_index] = True
                            score += 1
                            feedback_msg = "Perfect! Pose completed!"

                            # 播放成功语音和音效
                            try:
                                # 播放语音
                                speech_queue.put(POSE_SEQUENCE[current_pose_index]["success_speech"])
                                # 播放音效
                                sound_played = play_sound_effect(POSE_SEQUENCE[current_pose_index]["success_sound"])

                                # 显示点赞图片
                                if thumbs_up_image is not None and sound_played:
                                    thumbs_up_display_start = current_time
                                    thumbs_up_displaying = True
                            except Exception as e:
                                print(f"Error playing sound/speech: {e}")

                            # Move to next pose
                            next_pose_index = current_pose_index + 1
                            if next_pose_index < len(POSE_SEQUENCE):
                                current_pose_index = next_pose_index
                                current_pose = POSE_SEQUENCE[current_pose_index]

                                try:
                                    ref_pose = load_reference_pose(current_pose["ref_img"])
                                    target_image = cv2.imread(current_pose["target_img"], cv2.IMREAD_UNCHANGED)
                                    hint_image = cv2.imread(current_pose["hint_img"], cv2.IMREAD_UNCHANGED)
                                    target_image = cv2.resize(
                                        target_image,
                                        (int(screen_width * TARGET_IMG_SCALE),
                                         int(screen_width * TARGET_IMG_SCALE * target_image.shape[0] /
                                             target_image.shape[1]))
                                    )
                                except Exception as e:
                                    print(f"Error loading next pose: {e}")

                            is_matching = False
                            encouragement_played = False

                        # Play encouragement speech
                        elif duration > 2.0 and not encouragement_played:
                            play_encouragement()
                            encouragement_played = True
                else:
                    is_matching = False
                    encouragement_played = False
                    if missing_limbs:
                        feedback_msg = f"Missing limbs: {', '.join(missing_limbs)}"
                    else:
                        feedback_msg = f"Max deviation: {max_deviation:.1f}°"

                    show_hint = True

            # Display deviation info
            if last_deviation_text:
                draw_text_with_border(image, last_deviation_text, (20, h - 50), 0.7, 1, (50, 200, 255))

            # 显示提示图片（在绘制箭头之后） - 现在箭头会覆盖在提示图片上方
            if show_hint and not pose_completed[current_pose_index] and not thumbs_up_displaying:
                if max_deviation_joint:
                    joint_x, joint_y = max_deviation_joint
                    img_w = hint_img_resized.shape[1]
                    img_h = hint_img_resized.shape[0]

                    pos_x = joint_x - img_w // 2
                    pos_y = joint_y - img_h // 2

                    # Ensure position is within screen
                    pos_x = max(10, min(pos_x, w - img_w - 10))
                    pos_y = max(10, min(pos_y, h - img_h - 10))

                    # 现在先绘制提示图片
                    image = blend_transparent(image, hint_img_resized, pos_x, pos_y)

                    if max_deviation_limb:
                        text_size = cv2.getTextSize(max_deviation_limb, cv2.FONT_HERSHEY_SIMPLEX, 0.7, 1)[0]
                        text_x = pos_x + (img_w - text_size[0]) // 2
                        draw_text_with_border(image, max_deviation_limb, (text_x, pos_y - 10), 0.7, 1)
                else:
                    hint_x = (w - hint_img_resized.shape[1]) // 2
                    image = blend_transparent(image, hint_img_resized, hint_x, h - hint_img_resized.shape[0] - 20)

            # Display score (bottom center)
            draw_text_with_border(image, f"Score: {score}/{len(POSE_SEQUENCE)}", (w // 2 - 70, h - 30), 1.0, 2)

            # Display current pose info (top left)
            pose_info = f"Pose {current_pose_index + 1}/{len(POSE_SEQUENCE)}: {current_pose['name']}"
            draw_text_with_border(image, pose_info, (20, 40), 0.8, 2)

            # Display elapsed time (top right)
            elapsed_time = int(time.time() - start_time)
            time_text = f"Time: {elapsed_time // 60:02d}:{elapsed_time % 60:02d}"
            draw_text_with_border(image, time_text, (w - 200, 40), 0.8, 2)

            # 显示点赞图片（如果正在显示）- 居中显示
            if thumbs_up_displaying and thumbs_up_image is not None:
                current_time = time.time()
                display_duration = current_time - thumbs_up_display_start

                if display_duration < THUMBS_UP_DISPLAY_DURATION:
                    # 计算点赞图片位置（屏幕中央）
                    thumbs_up_x = (w - thumbs_up_image.shape[1]) // 2
                    thumbs_up_y = (h - thumbs_up_image.shape[0]) // 2

                    # 混合显示点赞图片
                    image = blend_transparent(image, thumbs_up_image, thumbs_up_x, thumbs_up_y)
                else:
                    thumbs_up_displaying = False

            # Display progress bar below the target image
            if is_matching and not pose_completed[current_pose_index]:
                duration = current_time - start_matching_time
                progress = min(duration / REQUIRED_DURATION, 1.0)

                # 进度条位置：在目标图像下方20像素处
                progress_bar_y = target_img_bottom + 20

                # 绘制进度条背景
                cv2.rectangle(image, (w // 2 - 100, progress_bar_y),
                              (w // 2 + 100, progress_bar_y + 10),
                              (50, 50, 50), -1)

                # 绘制进度条前景
                cv2.rectangle(image, (w // 2 - 100, progress_bar_y),
                              (int(w // 2 - 100 + 200 * progress), progress_bar_y + 10),
                              (0, 255, 0), -1)

                # 显示进度百分比 - 在进度条下方
                percent_text = f"{int(progress * 100)}%"
                draw_text_with_border(image, percent_text,
                                      (w // 2 - 20, progress_bar_y + 30),
                                      0.8, 2)

            # Check if all poses completed
            if all(pose_completed):
                draw_text_with_border(image, "All poses completed! Congratulations!",
                                      (w // 2 - 200, h // 2), 1.5, 3)

                # 播放最终祝贺语音和音效（确保只播放一次）
                if not final_congrats_played:
                    try:
                        # 播放最终语音
                        speech_queue.put("All poses completed! Congratulations!")
                        # 播放最终音效（使用最后一个姿势的音效）
                        sound_played = play_sound_effect(POSE_SEQUENCE[-1]["success_sound"])

                        # 显示点赞图片
                        if thumbs_up_image is not None and sound_played:
                            thumbs_up_display_start = time.time()
                            thumbs_up_displaying = True

                        final_congrats_played = True
                    except Exception as e:
                        print(f"Error playing final sound/speech: {e}")

            # Display frame
            cv2.imshow('Precision Pose Challenge', image)

            if cv2.waitKey(5) & 0xFF == ord('q'):
                break

    # Clean up resources
    speech_thread_running = False
    cap.release()
    cv2.destroyAllWindows()
    pygame.mixer.quit()  # 清理pygame资源


if __name__ == "__main__":
    main()