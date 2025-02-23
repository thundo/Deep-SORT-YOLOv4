#! /usr/bin/env python
# -*- coding: utf-8 -*-

from __future__ import division, print_function, absolute_import

from timeit import time
import warnings
import cv2
import numpy as np
from PIL import Image
from yolo import YOLO

from deep_sort import preprocessing
from deep_sort import nn_matching
from deep_sort.detection import Detection
from deep_sort.tracker import Tracker
from tools import generate_detections as gdet
import imutils.video
from videocaptureasync import VideoCaptureAsync
import click
from collections import OrderedDict

warnings.filterwarnings('ignore')

@click.command()
@click.option('--input', '-i', help='Input file [default: input/video.webm]', type=click.Path(exists=True), default='input/video.webm')
@click.option('--output', '-o', help='Output file [default: output/tracker.avi]', type=click.Path(), default='output/tracker.avi')
@click.option('--tracker', '-t', help='Cosine metric model [default: model_data/mars-small128.pb]', type=click.Path(exists=True), default='model_data/mars-small128.pb')
@click.option('--detect', '-d', multiple=True, help='Detect class [default: empty for all of them]')

def main(**config_kwargs):

    print(config_kwargs)

    yolo = YOLO(config_kwargs['detect'])

    # Definition of the parameters
    max_cosine_distance = 0.3
    nn_budget = None
    nms_max_overlap = 1.0

    # Deep SORT
    encoder = gdet.create_box_encoder(config_kwargs['tracker'], batch_size=1)

    metric = nn_matching.NearestNeighborDistanceMetric('cosine', max_cosine_distance, nn_budget)
    tracker = Tracker(metric)

    show_detections = True
    writeVideo_flag = True
    asyncVideo_flag = False
    frame_index = -1

    track_marker_radius = 8
    track_max_age = 50
    track_color = (0, 0, 255)
    text_color = (255, 255, 255)
    rect_color = (255, 255, 255)
    text_font = cv2.FONT_HERSHEY_DUPLEX
    text_line = cv2.LINE_AA

    if asyncVideo_flag:
        video_capture = VideoCaptureAsync(config_kwargs['input'])
    else:
        video_capture = cv2.VideoCapture(config_kwargs['input'])

    if asyncVideo_flag:
        video_capture.start()

    if writeVideo_flag:
        if asyncVideo_flag:
            w = int(video_capture.cap.get(3))
            h = int(video_capture.cap.get(4))
        else:
            w = int(video_capture.get(3))
            h = int(video_capture.get(4))
        fourcc = cv2.VideoWriter_fourcc(*'XVID')
        out = cv2.VideoWriter(config_kwargs['output'], fourcc, 30, (w, h))

    fps = 0.0
    fps_imutils = imutils.video.FPS().start()
    tracks_per_frame = OrderedDict()

    while True:
        ret, frame = video_capture.read()  # frame shape 640*480*3
        if ret != True:
            break
        frame_index += 1

        t1 = time.time()

        image = Image.fromarray(frame[..., ::-1])  # bgr to rgb
        boxes, confidence, classes = yolo.detect_image(image)

        features = encoder(frame, boxes)
        detections = [Detection(bbox, confidence, cls, feature) for bbox, confidence, cls, feature in
                      zip(boxes, confidence, classes, features)]

        # Run non-maxima suppression.
        boxes = np.array([d.tlwh for d in detections])
        scores = np.array([d.confidence for d in detections])
        classes = np.array([d.cls for d in detections])
        indices = preprocessing.non_max_suppression(boxes, nms_max_overlap, scores)
        detections = [detections[i] for i in indices]

        # Call the tracker
        tracker.predict()
        tracker.update(detections)

        for det in detections:
            bbox = det.to_tlbr()
            if show_detections and len(classes) > 0:
                det_cls = det.cls
                score = "%.2f" % (det.confidence * 100) + "%"
                cv2.putText(
                    img=frame,
                    text=str(det_cls) + " " + score,
                    org=(int(bbox[0]), int(bbox[3])),
                    fontFace=text_font,
                    fontScale=1e-3 * frame.shape[0],
                    color=text_color,
                    thickness=1,
                    lineType=text_line)
                cv2.rectangle(frame, (int(bbox[0]), int(bbox[1])), (int(bbox[2]), int(bbox[3])), (255, 0, 0), 2)

        tracks_per_frame[frame_index] = {}
        for track in tracker.tracks:
            if not track.is_confirmed() or track.time_since_update > 1:
                continue

            bbox = track.to_tlbr()
            center = track.center()
            tracks_per_frame[frame_index][track.track_id] = center

            adc = "%.2f" % (track.adc * 100) + "%"  # Average detection confidence
            cv2.rectangle(frame, (int(bbox[0]), int(bbox[1])), (int(bbox[2]), int(bbox[3])), rect_color, 2)
            cv2.putText(
                img=frame, 
                text= "ID: " + str(track.track_id), 
                org=(int(bbox[0]), int(bbox[1]) - 4), 
                fontFace=text_font,
                fontScale=1e-3 * frame.shape[0], 
                color=text_color, 
                thickness=1,
                lineType=text_line)
            if not show_detections:
                track_cls = track.cls
                cv2.putText(
                    img=frame,
                    text=str(track_cls),
                    org=(int(bbox[0] + 4), int(bbox[3]) - 4),
                    fontFace=text_font,
                    fontScale=1e-3 * frame.shape[0],
                    color=text_color,
                    thickness=1,
                    lineType=text_line)
                cv2.putText(
                    img=frame,
                    text='ADC: ' + adc,
                    org=(int(bbox[0]), int(bbox[3] + 2e-2 * frame.shape[1])),
                    fontFace=text_font,
                    fontScale=1e-3 * frame.shape[0],
                    color=text_color,
                    thickness=1,
                    lineType=text_line)

        # No tracks in this frame
        if not tracks_per_frame[frame_index]:
            del tracks_per_frame[frame_index]

        # Draw (fading) tracks
        for frame_idx in list(tracks_per_frame.keys()):
            # Delete tracks older than n frames
            frame_age = frame_index - frame_idx
            if frame_age > track_max_age:
                del tracks_per_frame[frame_idx]
                continue
            # Add overlay for alpha
            overlay = frame.copy()
            tracks = tracks_per_frame[frame_idx]
            for t_id, t_center in tracks.items():
                cv2.circle(overlay, (int(t_center[0]), int(t_center[1])), track_marker_radius, track_color, -1)

            alpha = 1 - frame_age / track_max_age
            frame = cv2.addWeighted(overlay, alpha, frame, 1 - alpha, 0)

        cv2.imshow('', frame)

        if writeVideo_flag:
            # save a frame
            out.write(frame)

        fps_imutils.update()

        if not asyncVideo_flag:
            fps = (fps + (1. / (time.time() - t1))) / 2
            print("FPS = %f" % (fps))

        # Press Q to stop!
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    fps_imutils.stop()
    print('imutils FPS: {}'.format(fps_imutils.fps()))

    if asyncVideo_flag:
        video_capture.stop()
    else:
        video_capture.release()

    if writeVideo_flag:
        out.release()

    cv2.destroyAllWindows()

if __name__ == '__main__':
    main()
