# -*- coding: utf-8 -*-
import cv2

#sx, syは線の始まりの位置
sx, sy = 0, 0
x, y = 0, 0
class mouseParam:
    def __init__(self, input_img_name):
        #マウス入力用のパラメータ
        self.mouseEvent = {"x":None, "y":None, "event":None, "flags":None}
        #マウス入力の設定
        cv2.setMouseCallback(input_img_name, self.__CallBackFunc, None)

    #コールバック関数
    def __CallBackFunc(self, eventType, x, y, flags, userdata):

        self.mouseEvent["x"] = x
        self.mouseEvent["y"] = y
        self.mouseEvent["event"] = eventType
        self.mouseEvent["flags"] = flags

    #マウス入力用のパラメータを返すための関数
    def getData(self):
        return self.mouseEvent

    #マウスイベントを返す関数
    def getEvent(self):
        return self.mouseEvent["event"]

    #マウスフラグを返す関数
    def getFlags(self):
        return self.mouseEvent["flags"]

    #xとyの座標を返す関数
    def getPos(self):
        return (self.mouseEvent["x"], self.mouseEvent["y"])

cap = cv2.VideoCapture(0)
click_points = []
draw = False
#コールバックの設定
window_name = "main window"

while True:
    # VideoCaptureから1フレーム読み込む
    ret, frame = cap.read()

    # 描画する
    cv2.line(frame, (x, y), (sx, sy), (0, 0, 255), 5)
    
    cv2.imshow(window_name, frame)

    # 描画結果の中でマウスの状態を取得する
    mouseData = mouseParam(window_name)

    # キー入力を1ms待って、k が27（ESC）だったらBreakする
    k = cv2.waitKey(1)
    if k == 27:
        break

    #左クリックがあったら表示
    if mouseData.getEvent() == cv2.EVENT_LBUTTONDOWN: # 左ボタンを押下したとき
        draw = True
        # global x, y
        x, y = mouseData.getPos()
        print("始点1={},{}".format(x,y))
    if mouseData.getEvent() == cv2.EVENT_LBUTTONUP: # 左ボタンを上げたとき
        draw = False
    if mouseData.getEvent() == cv2.EVENT_MOUSEMOVE and draw: # マウスが動いた時
        print("始点2={},{}".format(x,y))
        print(mouseData.getPos())
        sx, sy = mouseData.getPos()

cap.release()
cv2.destroyAllWindows()