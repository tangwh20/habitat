docker run -itd --rm --network=host --gpus=all -e DISPLAY=$DISPLAY \
  --ipc=host \
  -e http_proxy=$http_proxy \
  -e https_proxy=$https_proxy \
  -e all_proxy=$all_proxy \
  -e OPENAI_API_KEY=$OPENAI_API_KEY \
  -e TOKENIZERS_PARALLELISM=false \
  -v /home/yufeng/.Xauthority:/root/.Xauthority \
  -v /home/yufeng/:/home/yufeng/ \
  -v /media/yufeng:/media/yufeng \
  -v /media/yufeng/checkpoints:/root/.cache/huggingface/hub \
  --name openvla \
  aliciaji/openvla bash