import gi
gi.require_version('Gst', '1.0')
from gi.repository import GObject, Gst, GLib
import cv2
import numpy as np
import os
from common.bus_call import bus_call

import time
import pyds

Gst.init(None)

all_detections = [] 

def get_jpg_files(root_folder):
    jpg_files = []
    
    # Procházení složky UAV-benchmark-M
    for folder in os.listdir(root_folder):
        folder_path = os.path.join(root_folder, folder)
        
        # Kontrola, jestli je složka ve formátu MXXXX
        if os.path.isdir(folder_path) and folder.startswith('M') and folder[1:].isdigit() and len(folder) == 5:
            # Procházení všech souborů v podsložce
            for file in os.listdir(folder_path):
                if file.endswith('.jpg'):
                    jpg_files.append((os.path.join(folder_path, file), file, folder_path.split("/")[-1]))
    return jpg_files

def test_pad_probe(pad, info, u_data):
    #print("Testovaci probe: buffer prosel timto padem.")
    return Gst.PadProbeReturn.OK

def osd_sink_pad_buffer_probe(pad, info, u_data):
    global all_detections
    gst_buffer = info.get_buffer()
    if not gst_buffer:
        print("Chyba při získávání bufferu")
        return Gst.PadProbeReturn.OK
    
    # Získání batch metadat
    batch_meta = pyds.gst_buffer_get_nvds_batch_meta(hash(gst_buffer))
    if not batch_meta:
        print("Chyba: batch_meta neni k dispozici.")
        return Gst.PadProbeReturn.OK

    # Iterace přes frame metadaty
    l_frame = batch_meta.frame_meta_list
    while l_frame is not None:
        try:
            frame_meta = pyds.NvDsFrameMeta.cast(l_frame.data)
        except StopIteration:
            break

        obj_meta_list = frame_meta.obj_meta_list
        detections = []
        
        # Iterace přes objekty v aktuálním snímku
        while obj_meta_list is not None:
            try:
                obj_meta = pyds.NvDsObjectMeta.cast(obj_meta_list.data)
                detections.append({
                    "class_id": obj_meta.class_id,
                    "confidence": obj_meta.confidence,
                    "bbox": [
                        obj_meta.rect_params.left,
                        obj_meta.rect_params.top,
                        obj_meta.rect_params.width,
                        obj_meta.rect_params.height
                    ]
                })
                obj_meta_list = obj_meta_list.next
            except StopIteration:
                break

        print()
        print("Detekce:", detections)
        print()
        all_detections.append(detections)
        # Přejít na další frame v seznamu
        l_frame = l_frame.next
    
    return Gst.PadProbeReturn.OK

def process_image(appsrc, image_path):
    #print("Posalim obrazek do appsrc")  # Ladící zpráva
    print(image_path)
    image = cv2.imread(image_path)
    if image is None:
        print(f"Chyba pri nacitani obrazku: {image_path}")
        return
    
    # Změna velikosti a převod na RGBA
    image_resized = cv2.resize(image, (960, 544))
    image_rgba = cv2.cvtColor(image_resized, cv2.COLOR_BGR2RGBA)
    image_np = np.array(image_rgba, dtype=np.uint8)
    
    gst_buffer = Gst.Buffer.new_allocate(None, image_np.nbytes, None)
    gst_buffer.fill(0, image_np.tobytes())
    appsrc.emit("push-buffer", gst_buffer)
    #print("Obrazek byl odeslan do pipeline.")

def bus_call(bus, message, loop):
    if message.type == Gst.MessageType.EOS:
        print("End-Of-Stream reached.")
        loop.quit()
    elif message.type == Gst.MessageType.ERROR:
        err, debug = message.parse_error()
        print(f"Error: {err}, {debug}")
        loop.quit()
    return True

def main():
    pipeline = Gst.parse_launch(
        'appsrc name=src ! '
        'video/x-raw, format=RGBA, width=960, height=544, framerate=1/1 ! '
        'nvvideoconvert ! '
        'video/x-raw(memory:NVMM), format=RGBA, width=960, height=544 ! '
        'mux.sink_0 nvstreammux name=mux batch-size=1 width=960 height=544 ! '
        'nvinfer config-file-path=deepstream_app_config.txt name=nv ! '
        'nvdsosd name=nvosd ! '
        'fakesink'
    )
    
    appsrc = pipeline.get_by_name("src")

    if appsrc:
        src_pad = appsrc.get_static_pad("src")
        if src_pad:
            src_pad.add_probe(Gst.PadProbeType.BUFFER, test_pad_probe, 0)
        else:
            print("Chyba: Nelze získat src pad appsrc")
    else:
        print("Chyba: Nelze získat element appsrc")

    nvosd = pipeline.get_by_name("nvosd")
    if appsrc:
        sink_pad = nvosd.get_static_pad("sink")
        if sink_pad:
            sink_pad.add_probe(Gst.PadProbeType.BUFFER, osd_sink_pad_buffer_probe, 0)
        else:
            print("Chyba: Nelze získat src pad appsrc")
    else:
        print("Chyba: Nelze získat element appsrc")

    mux = pipeline.get_by_name('mux')
    mux.set_property("nvbuf-memory-type", int(pyds.NVBUF_MEM_CUDA_DEVICE))

    # Spuštění pipeline
    pipeline.set_state(Gst.State.PLAYING)
    print("Pipeline running.")
    
    # Inicializace GLib smyčky
    loop = GObject.MainLoop()
    bus = pipeline.get_bus()
    bus.add_signal_watch()
    bus.connect ("message", bus_call, loop)
    
    # Spuštění posílání obrázků v samostatném vlákně
    def push_images():
        time.sleep(0.5) 
        dataset_path = '/root/uavdt/'
        image_files = get_jpg_files(dataset_path)
        #image_files = [os.path.join(dataset_path, f) for f in os.listdir(dataset_path) if f.endswith('.jpg')]
        for image_path in image_files:
            process_image(appsrc, image_path[0])
            time.sleep(0.02)  # Pauza, aby měl pipeline čas na zpracování obrázku
        appsrc.emit("end-of-stream")
        exit(0)
    
    from threading import Thread
    image_thread = Thread(target=push_images)
    image_thread.start()
    
    try:
        print(all_detections)
        loop.run()
    except Exception as e:
        print(e)
    finally:
        pipeline.set_state(Gst.State.NULL)

if __name__ == '__main__':
    main()