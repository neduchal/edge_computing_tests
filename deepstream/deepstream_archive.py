import cv2
import numpy as np
import pyds
import gi
import os
import time
import sys
gi.require_version('Gst', '1.0')
from gi.repository import GObject, Gst, GLib
from common.bus_call import bus_call

def numpy_to_gst_buffer(image_np):

    h, w, c = image_np.shape
    gst_buffer = Gst.Buffer.new_allocate(None, image_np.nbytes, None)
    caps = Gst.Caps.from_string("video/x-raw, format=RGBA, width={}, height={}, framerate=1/1".format(w, h))
    

    success, map_info = gst_buffer.map(Gst.MapFlags.WRITE)
    if success:
        # Zkopírování dat z numpy pole jako bytes do map_info.data
        map_info_memory = bytearray(map_info.data)
        map_info_memory[:image_np.nbytes] = image_np.tobytes()
        gst_buffer.unmap(map_info)  # Ruční odmapování po skončení

    return gst_buffer, caps

""" def numpy_to_gst_buffer(array: np.ndarray) -> Gst.Buffer:

    return Gst.Buffer.new_wrapped(array.tobytes()) """

def process_image(appsrc, image_path):
    """Process single image through DeepStream pipeline."""
    # Načíst obrázek pomocí OpenCV
    image = cv2.imread(image_path)
    if image is None:
        print(f"Chyba pri nacitani obrazku: {image_path}")
        return
    
    image_resized = cv2.resize(image, (960, 544))

    # Transformace obrazu do RGB formátu a do numpy pole
    image_rgb = cv2.cvtColor(image_resized, cv2.COLOR_BGR2BGRA)
    image_np = np.array(image_rgb, dtype=np.uint8)

    print(image_np.shape)
    
    # Převod numpy obrazu na Gst.Buffer
    gst_buffer, caps = numpy_to_gst_buffer(image_np)
    print(caps)
    
    # Nastavení caps (pouze jednou, pokud se rozlišení nemění)
    appsrc.set_property("caps", caps)
    
    # Poslat obrázek do pipeline
    appsrc.emit("push-buffer", gst_buffer)
    print(f"Zpracovavam obrazek: {image_path}")

def osd_sink_pad_buffer_probe(pad, info, u_data):
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

        print("Detekce:", detections)

        # Přejít na další frame v seznamu
        l_frame = l_frame.next
    
    return Gst.PadProbeReturn.OK

# Callback funkce pro extrakci výsledků detekce z `nvinfer`
def nvinfer_src_pad_buffer_probe(pad, info, u_data):
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

        print("Detekce:", detections)

        # Přejít na další frame v seznamu
        l_frame = l_frame.next
    
    return Gst.PadProbeReturn.OK

def appsrc_pad_buffer_probe(pad, info, u_data):
    print("Juhu")
    return Gst.PadProbeReturn.OK

def main():

    GObject.threads_init()
    Gst.init(None)

    # Nastavení pipeline
    pipeline = Gst.parse_launch(
        'appsrc name=src ! '
        'video/x-raw, format=RGBA, width=960, height=544, framerate=1/1 ! '
        'nvvideoconvert name=nvc ! '
        'video/x-raw(memory:NVMM), format=RGBA, width=960, height=544, framerate=1/1 ! '
        'mux.sink_0 nvstreammux name=mux batch-size=1 width=960 height=544 ! '
        'nvinfer name=nv config-file-path=/opt/nvidia/deepstream/deepstream-6.0/samples/models/tao_pretrained_models/Projekty/EdgeComputing/detektory/deepstream/deepstream_app_config.txt ! '
        'nvdsosd name=nvosd ! ' 
        'appsink'
    )
    
    # Nastavení zdroje appsrc
    appsrc = pipeline.get_by_name('src')
    #caps = Gst.Caps.from_string("video/x-raw, format=(string)RGB, width=(int)960, height=(int)544, framerate=(fraction)1/1")
    #caps = Gst.Caps.from_string("video/x-raw(memory:NVMM), format=RGBA, width=(int)960, height=(int)544, framerate=(fraction)1/1")
    caps = Gst.Caps.from_string("video/x-raw, format=RGBA, width=960, height=544, framerate=1/1")

    appsrc.set_property("caps", caps)    
    appsrc.set_property("format", Gst.Format.TIME)

    # Připojení `appsrc` do `nvstreammux`
    mux = pipeline.get_by_name('mux')
    mux.set_property("nvbuf-memory-type", int(pyds.NVBUF_MEM_CUDA_DEVICE))
    #src_pad = appsrc.get_static_pad('src')
    #sink_pad = mux.get_request_pad('sink_0')
    #src_pad.link(sink_pad)

    appsrcpad = appsrc.get_static_pad("src")
    appsrcpad.add_probe(Gst.PadProbeType.BUFFER, appsrc_pad_buffer_probe, 0)

    nvc = pipeline.get_by_name("nvc") 
    nvcpad = nvc.get_static_pad("src")
    nvcpad.add_probe(Gst.PadProbeType.BUFFER, appsrc_pad_buffer_probe, 0)

    nvm = pipeline.get_by_name("mux") 
    nvmpad = nvm.get_static_pad("src")
    nvmpad.add_probe(Gst.PadProbeType.BUFFER, appsrc_pad_buffer_probe, 0)

    """ 
    nvinfer = pipeline.get_by_name("nv")  # Ujistěte se, že jméno odpovídá vašemu nastavení
    if nvinfer:
        src_pad = nvinfer.get_static_pad("src")
        if src_pad:
            src_pad.add_probe(Gst.PadProbeType.BUFFER, nvinfer_src_pad_buffer_probe, 0)
            print("Pridan src pad nvinfer")
        else:
            print("Chyba: Nelze získat src pad nvinfer")
    else:
        print("Chyba: Nelze získat element nvinfer") 

    # Nastavení callbacku na výstupním padu
    sink_pad = pipeline.get_by_name('nvosd').get_static_pad('sink')
    if not sink_pad:
        print("Error: Unable to get sink pad of nvosd")
    else:
        print("Sink pad obtained successfully")
        sink_pad.add_probe(Gst.PadProbeType.BUFFER, osd_sink_pad_buffer_probe, 0)
    # Spuštění pipeline
    # create an event loop and feed gstreamer bus mesages to it
    """
    loop = GObject.MainLoop()
    bus = pipeline.get_bus()
    bus.add_signal_watch()
    bus.connect ("message", bus_call, loop) 
    



    pipeline.set_state(Gst.State.PLAYING)

    # Cesta k datasetu obrázků
    dataset_path = '/root/archive/'
    image_files = [os.path.join(dataset_path, f) for f in os.listdir(dataset_path) if f.endswith('.jpg')]

    # Zpracování všech obrázků v datasetu
    for image_path in image_files[0:5]:
        process_image(appsrc, image_path)
        # Krátká pauza, aby pipeline stihla zpracovat jeden snímek před odesláním dalšího
        time.sleep(0.1)

    # Ukončení streamu
    appsrc.emit("end-of-stream")
    
    try:
        #loop = GLib.MainLoop()   
        print("BLA")    
        loop.run()
        print("BLA")    

    except Exception as e:
        print("BLO")
        print(e)
    finally:
        pipeline.set_state(Gst.State.NULL)
        print("BLE")

if __name__ == '__main__':
    main()
