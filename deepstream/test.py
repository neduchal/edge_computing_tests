import cv2
import numpy as np
import pyds
import gi
import os
import time
gi.require_version('Gst', '1.0')
from gi.repository import Gst, GLib

Gst.init(None)

def numpy_to_gst_buffer(image_np):
    """Convert numpy array to Gst.Buffer"""
    h, w, c = image_np.shape
    gst_buffer = Gst.Buffer.new_allocate(None, image_np.nbytes, None)
    
    # Namapování bufferu a přímé zkopírování dat
    success, map_info = gst_buffer.map(Gst.MapFlags.WRITE)
    if success:
        map_info.data = image_np.tobytes()
        gst_buffer.unmap(map_info)  # Ruční odmapování po skončení

    return gst_buffer

def process_image(appsrc, image_path):
    """Process single image through DeepStream pipeline."""
    image = cv2.imread(image_path)
    if image is None:
        print(f"Chyba pri nacitani obrazku: {image_path}")
        return
    
    # Změna velikosti a převod na BGRx
    image_resized = cv2.resize(image, (960, 544))
    image_bgrx = cv2.cvtColor(image_resized, cv2.COLOR_BGR2BGRA)
    image_np = np.array(image_bgrx, dtype=np.uint8)
    
    gst_buffer = numpy_to_gst_buffer(image_np)
    appsrc.emit("push-buffer", gst_buffer)
    print(f"Zpracovavam obrazek: {image_path}")

def main():
    # Nastavení pipeline s použitím `nvstreammux`
    pipeline = Gst.parse_launch(
        'appsrc name=src ! '
        'video/x-raw, format=BGRx, width=960, height=544, framerate=1/1 ! '
        'nvvideoconvert ! '
        'video/x-raw(memory:NVMM), format=NV12, width=960, height=544 ! '
        'nvstreammux name=mux batch-size=1 width=960 height=544 ! '
        'nvinfer config-file-path=/opt/nvidia/deepstream/deepstream-6.0/samples/models/tao_pretrained_models/Projekty/EdgeComputing/detektory/deepstream/deepstream_app_config.txt ! '
        'nvdsosd name=nvosd ! '
        'appsink'
    )
    
    # Získání `appsrc` a nastavení `caps`
    appsrc = pipeline.get_by_name('src')
    caps = Gst.Caps.from_string("video/x-raw, format=BGRx, width=960, height=544, framerate=1/1")
    appsrc.set_property("caps", caps)
    appsrc.set_property("format", Gst.Format.TIME)
    
    # Získání `nvstreammux` a propojení padů
    mux = pipeline.get_by_name('mux')
    sink_pad = mux.get_request_pad("sink_0")  # Žádost o pad 'sink_0' na 'nvstreammux'
    src_pad = appsrc.get_static_pad("src")
    src_pad.link(sink_pad)

    # Nastavení callbacku pro detekce (zbytek kódu zůstává stejný)
    sink_pad_osd = pipeline.get_by_name('nvosd').get_static_pad('sink')
    if not sink_pad_osd:
        print("Error: Unable to get sink pad of nvosd")
    else:
        print("Sink pad obtained successfully")
        sink_pad_osd.add_probe(Gst.PadProbeType.BUFFER, osd_sink_pad_buffer_probe, 0)

    # Spuštění pipeline
    pipeline.set_state(Gst.State.PLAYING)
    
    # Spuštění smyčky a odeslání obrázků
    try:
        loop = GLib.MainLoop()
        from threading import Thread

        def push_images():
            dataset_path = '/root/archive/'
            image_files = [os.path.join(dataset_path, f) for f in os.listdir(dataset_path) if f.endswith('.jpg')]
            for image_path in image_files[0:5]:
                process_image(appsrc, image_path)
            appsrc.emit("end-of-stream")
        
        image_thread = Thread(target=push_images)
        image_thread.start()
        
        loop.run()
    except Exception as e:
        print(e)
    finally:
        pipeline.set_state(Gst.State.NULL)

if __name__ == '__main__':
    main()