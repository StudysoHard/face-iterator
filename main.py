# from flask_script import Manager
from controller import create_app
import requests

parser = argparse.ArgumentParser(description="subprocess task")
parser.add_argument('--port', type=int, default=0, help='set task run prot')
parser.add_argument('--streamUrl', type=int, default=0, help='run url')
args = parser.parse_args()
inPort = args.port
streamUrl = args.streamurl

# 创建APP对象
app = create_app('dev')
# # 创建脚本管理


if __name__ == '__main__':
    # mgr.run()
    app.run(threaded=True, host="localhost",port=inPort)



