import requests
import sys
from pathlib import Path



if(len(sys.argv) < 4):
    print("this script needs 3 parameters : model.xml model.bin compiled_model.blob")
else:
    xml_path = str(Path(sys.argv[1]).resolve().absolute())
    bin_path = str(Path(sys.argv[2]).resolve().absolute())
    blob_path = str(Path(sys.argv[3]).resolve().absolute())

    # Check file types
    if(not xml_path.endswith(".xml")):
        print("1st parameter must be a xml file (from OpenVino 2020.1 IR)")
        exit()
    if(not bin_path.endswith(".bin")):
        print("2nd parameter must be a bin file (from OpenVino 2020.1 IR)")
        exit()
    if(not blob_path.endswith(".blob")):
        print("3rd parameter must be a blob file (path to your future compiled model)")
        exit()
    
    # Check if file exists
    try:
        xml_file = open(xml_path, 'rb')
    except:
        print("xml file doesn't exist")
        exit()
    try:
        bin_file = open(bin_path, 'rb')
    except:
        print("bin file doesn't exist")
        exit()
    
    
    # Build a request to the DepthAI model compiler API : (.bin, .xml) --> .blob
    url = "http://69.164.214.171:8083/compile"
    payload = {
        'compiler_params': '-ip U8 -VPU_NUMBER_OF_SHAVES 8 -VPU_NUMBER_OF_CMX_SLICES 8',
        'compile_type': 'myriad'
    }
    files = {
        'definition': xml_file,
        'weights': bin_file
    }
    params = {
        'version': '2021.1'
    }

    # Download the blob file and write it
    response = requests.post(url, data=payload, files=files, params=params)
    blobnameraw = response.headers.get('Content-Disposition')
    blobname = blobnameraw[blobnameraw.find('='):][1:]
    with open(blob_path, 'wb') as f:
        f.write(response.content)