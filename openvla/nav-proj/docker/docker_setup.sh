# script to run in docker container to setup the environmet

# build project package
source /env/bin/activate
cd /home/yufeng/homebot_real
pip install -e .
cd /home/yufeng/openvla
pip install -e .

# set shortcut command in .bashrc
tee -a /root/.bashrc << END
# shortcut command for entering project
function openvla() {
    source /env/bin/activate
    cd /home/yufeng/openvla
}
# Connect to local VPN
function set_proxy() {
  export http_proxy="http://127.0.0.1:1081" https_proxy="socks5://127.0.0.1:1080" all_proxy="socks5://127.0.0.1:1080"
}
function unset_proxy() {
  unset http_proxy https_proxy all_proxy
}
# tensorflow directory
export TFDS_DATA_DIR="/media/yufeng/tensorflow_dataset"
END
source /root/.bashrc