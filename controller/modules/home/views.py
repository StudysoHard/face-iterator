from flask import session, render_template, redirect, url_for, Response, make_response,request
from controller.modules.home import home_blu
from controller.utils.camera import VideoCamera
from controller.utils.mysql import OperationMysql
from controller.utils import  milvus

video_camera = None
global_frame = None
flag = True
# 主页
# 主页
@home_blu.route('/')
def index():
    # 模板渲染
    username = session.get("username")
    ambient =round(28.88,1)#(sensor.get_ambient(),1)
    temp =round(36.55,1)#(sensor.get_object_1(),1)
    tempInfo = {
        'ambient' : ambient,
        'temp'    : temp
    }
    #bus.close()
    if not username:
        return render_template("test.html",**tempInfo)
    return render_template("index.html",**tempInfo)

# 视频流
@home_blu.route('/video_viewer')
def video_viewer():
    return Response(video_stream(),
                    mimetype='multipart/x-mixed-replace; boundary=frame')


@home_blu.route('/camera_change',methods=["GET"])
def camera_change():
    url = request.args['url']
    video_camera.set(False,url)
    return "OK"


@home_blu.route('/read_camera')
def read_camera():
    # sql =  OperationMysql()
    # result =  sql.search_all_camera()
    return "hello"

@home_blu.route('/black_image', methods=['POST'])
def black_image():
    req = request.get_json()

    if "img" in list(req.keys()):
        raw_content = req["img"]  # list

        for item in raw_content:  # item is in type of dict
            instance = []
            img1 = item["img1"]
            img2 = item["img2"]

        milvus.insertXingGuanCollection(img1)


# 获取视频流
def video_stream():
    global video_camera
    global global_frame
    video_camera = VideoCamera(0,True)

    while True:
        frame = video_camera.get_frame()
        if frame is not None:
            global_frame = frame
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n\r\n')
        else:
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + global_frame + b'\r\n\r\n')



