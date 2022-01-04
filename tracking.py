import cv2
import json


def get_1st_rect(path):
    with open(path, 'rb') as f:
        ann = json.load(f)
    frm_1st = ann['sequence'][0]
    rect = (int(frm_1st["TgtXPos_LeftUp"]), int(frm_1st["TgtYPos_LeftUp"]),
            int(frm_1st["TgtWidth"]), int(frm_1st["TgtHeight"]))
    return rect


def tracking(tracker, video_path, ann_path):
    # 動画読み込み
    cap = cv2.VideoCapture(video_path)
    # アノテーションファイルを読み込み、最初の矩形を取得
    bbox = get_1st_rect(ann_path)

    # 最初のフレームを取得＆最初のbbox情報を用いてtrackerを初期化
    while True:
        ret, frame = cap.read()
        if not ret:
            continue
        # # ROIを手動選択する場合
        # bbox = (0, 0, 10, 10)
        # bbox = cv2.selectROI(frame, False)
        tracker.init(frame, bbox)
        tracker.update(frame)
        # cv2.destroyAllWindows()
        break

    while True:
        # VideoCaptureから1フレーム読み込む
        ret, frame = cap.read()
        # 空のフレームがあった場合 or 動画の末尾に到達した場合
        if not ret:
            k = cv2.waitKey(1)
            if k == 27:     # ESCキーが押されたら終了
                break
            continue

        # Start timer
        timer = cv2.getTickCount()

        # トラッカーをアップデートする
        track, bbox = tracker.update(frame)

        # FPSを計算する
        fps = cv2.getTickFrequency() / (cv2.getTickCount() - timer)

        # 検出した場所に四角を書く
        if track:
            # Tracking success
            p1 = (int(bbox[0]), int(bbox[1]))
            p2 = (int(bbox[0] + bbox[2]), int(bbox[1] + bbox[3]))
            cv2.rectangle(frame, p1, p2, (0,255,0), 2, 1)
        else :
            # トラッキングが外れたら警告を表示する
            cv2.putText(frame, "Failure", (10,50), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,255,0), 1, cv2.LINE_AA);

        # FPSを表示する
        cv2.putText(frame, "FPS : " + str(int(fps)), (10,20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,255,0), 1, cv2.LINE_AA);

        # 加工済の画像を表示する
        cv2.imshow("Tracking", frame)

        # ESCキーが押されたらbreak
        k = cv2.waitKey(1)
        if k == 27:
            break

    # キャプチャをリリースして、ウィンドウをすべて閉じる
    cap.release()
    cv2.destroyAllWindows()


if __name__ == '__main__':
    """
    Tracking手法を選ぶ。適当にコメントアウトして実行する。
    """
    # Boosting
    # tracker = cv2.TrackerBoosting_create()

    # MIL
    # tracker = cv2.TrackerMIL_create()

    # KCF
    tracker = cv2.TrackerKCF_create()

    # TLD #GPUコンパイラのエラーが出ているっぽい
    # tracker = cv2.TrackerTLD_create()

    # MedianFlow
    # tracker = cv2.TrackerMedianFlow_create()

    # GOTURN # モデルが無いよって怒られた
    # https://github.com/opencv/opencv_contrib/issues/941#issuecomment-343384500
    # https://github.com/Auron-X/GOTURN-Example
    # http://cs.stanford.edu/people/davheld/public/GOTURN/trained_model/tracker.caffemodel
    # tracker = cv2.TrackerGOTURN_create()

    # シーンごとに処理
    n_scene = 20
    for scene_num in range(n_scene):
        # 動画読み込み
        video_path = "../train_videos/" + str(scene_num).zfill(3) + "/Right.mp4"
        ann_path = "../train_annotations/" + str(scene_num).zfill(3) + ".json"
        tracking(tracker, video_path, ann_path)
