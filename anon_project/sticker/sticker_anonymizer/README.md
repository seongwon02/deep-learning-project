## 사용법
[1]
!apt-get -y install fonts-noto-color-emoji
!pip install ultralytics

[2]
%cd deep-learning-project/face-anonymizer/sticker_mode

[3]
!python extract_detections.py \
    --input input.mp4 \
    --model yolo11s-pose_widerface.pt \
    --output detections.json

[4]
!python -m sticker_anonymizer \
    --input input.mp4 \
    --output result.mp4 \
    --detections detections.json \
    --emoji "💀"

# 참고: 4단계의 --emoji에는 다른 이모지 넣어 사용해도 됩니다.
ex. 🤖 🐻 🐱 🐶 🐰 🦊 🐼 🐯 🐵 🐸 🐷 😀 😎 🥸 🤡 👽 👾 💀 🤠