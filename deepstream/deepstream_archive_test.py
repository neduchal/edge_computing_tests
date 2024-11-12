#!/usr/bin/env python
import json
import logging
import cv2


import sys
sys.path.append('../')
import gi
gi.require_version('Gst', '1.0')
from gi.repository import GObject, Gst
from common.bus_call import bus_call
import numpy as np
import pyds
import time

def ndarray_to_gst_buffer(array: np.ndarray) -> Gst.Buffer:
    """Converts numpy array to Gst.Buffer"""
    return Gst.Buffer.new_wrapped(array.tobytes())


def main():
    # Standard GStreamer initialization
    GObject.threads_init()
    Gst.init(None)

    # Create Pipeline element that will form a connection of other elements
    print("Creating Pipeline \n ")
    pipeline = Gst.Pipeline()

    if not pipeline:
        sys.stderr.write(" Unable to create Pipeline \n")

    # Source element for reading from the file
    print("Creating Source \n ")
    appsource = Gst.ElementFactory.make("appsrc", "numpy-source")
    if not appsource:
        sys.stderr.write(" Unable to create Source \n")

    nvvideoconvert = Gst.ElementFactory.make("nvvideoconvert","nv-videoconv")
    if not nvvideoconvert:
        sys.stderr.write(" error nvvid1")

    caps_filter = Gst.ElementFactory.make("capsfilter","capsfilter1")
    if not caps_filter:
        sys.stderr.write(" error capsf1")

    transform = Gst.ElementFactory.make("nvegltransform", "nvegl-transform")

    sink = Gst.ElementFactory.make("nveglglessink", "nvvideo-renderer")
    if not sink:
        sys.stderr.write(" Unable to create egl sink \n")

    caps_in = Gst.Caps.from_string("video/x-raw,format=RGBA,width=640,height=480,framerate=30/1")
    caps = Gst.Caps.from_string("video/x-raw(memory:NVMM),format=NV12,width=640,height=480,framerate=30/1")
    appsource.set_property('caps', caps_in)
    caps_filter.set_property('caps',caps)

    print("Adding elements to Pipeline \n")
    pipeline.add(appsource)
    pipeline.add(nvvideoconvert)
    pipeline.add(caps_filter)
    pipeline.add(transform)
    pipeline.add(sink)
    
    # Working Link pipeline
    print("Linking elements in the Pipeline \n")
    appsource.link(nvvideoconvert)
    nvvideoconvert.link(caps_filter)
    caps_filter.link(transform)
    transform.link(sink)

    # create an event loop and feed gstreamer bus mesages to it
    loop = GObject.MainLoop()
    bus = pipeline.get_bus()
    bus.add_signal_watch()
    bus.connect ("message", bus_call, loop)


    # start play back and listen to events
    print("Starting pipeline \n")
    pipeline.set_state(Gst.State.PLAYING)

    # Push buffer and check
    for _ in range(10):
        arr = np.random.randint(low=0,high=255,size=(480,640,3),dtype=np.uint8)
        arr = cv2.cvtColor(arr, cv2.COLOR_BGR2RGBA)
        appsource.emit("push-buffer", ndarray_to_gst_buffer(arr))
        time.sleep(0.3)
    appsource.emit("end-of-stream")
    try:
        loop.run()
    except:
        pass
    # cleanup
    pipeline.set_state(Gst.State.NULL)

if __name__ == '__main__':
    sys.exit(main())