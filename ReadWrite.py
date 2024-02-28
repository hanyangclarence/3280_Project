import struct
import numpy
import os


# write the contents of data to filepath
# returns 0 on success, return 1 and print error message if failed
# data is frames[]
def write_wav(data, filepath, rate=44100, channels=2, bytes_per_sample=2):
    if len(data)==0:
        print("No data to write")
        return
    try:
        file = open(filepath, "wb")
        
        datasize = len(data) * len(data[0])   # every frame element seems to have length 4096
        if channels==1:
            datasize *=2
        
        file.write("RIFF".encode())                     # 0-3    RIFF
        file.write(struct.pack('i', 36+datasize))       # 4-7    chunksize = datasize + 36
        file.write("WAVEfmt ".encode())                 # 8-15   WAVEfmt(SPACE)
        file.write(struct.pack('i', 16))                # 16-19  SubchunkSize = 16
        file.write(struct.pack('h', 1))                 # 20-21  AudioFormat = 1
        file.write(struct.pack('h', 2))                 # 22-23  NumOfChannels always 2
        file.write(struct.pack('i', rate))              # 24-27  SampleRate
        byte_rate = rate * 2 * bytes_per_sample
        file.write(struct.pack('i', byte_rate))         # 28-31  ByteRate
        block_align = 2 * bytes_per_sample
        file.write(struct.pack('h', block_align))       # 32-33  BlockAlign
        bits_per_sample = bytes_per_sample * 8
        file.write(struct.pack('h', bits_per_sample))   # 34-35  BitsPerSample
        file.write("data".encode())                     # 36-39  data
        file.write(struct.pack('i', datasize))          # 40-43  datasize
        
        if channels==2:
            for d in data:
                file.write(d)
        elif channels==1:
            for d in data:
                for i in range(0, len(d), bytes_per_sample):
                    file.write(d[i:i+bytes_per_sample])
                    file.write(d[i:i+bytes_per_sample])
        
        file.close()
        print("Write success")
        return 0
    except Exception as e:
        os.remove(filepath)
        print("Error during write")
        print(e)
        return 1


# reads wav from filepath, and returns a tuple (frames[], sample_rate, channels, bytes_per_sample)
# returns 1 and prints error message if failed
# remember to check the return value!
def read_wav(filepath, frame_size=4096):
    try:
        file = open(filepath, "rb")
        
        assert str(file.read(4), encoding='utf-8')=="RIFF", "File does not start with RIFF"         # 0-3    RIFF
        chunksize = int.from_bytes(file.read(4), "little")                                          # 4-7    chunksize = datasize + 36
        assert str(file.read(8), encoding='utf-8')=="WAVEfmt ", "File has incorrect WAVEfmt part"   # 8-15   WAVEfmt(SPACE)
        assert int.from_bytes(file.read(4), "little")==16, "SubchunkSize is not 16"                 # 16-19  SubchunkSize = 16
        assert int.from_bytes(file.read(2), "little")==1, "AudioFormat is not 1"                    # 20-21  AudioFormat = 1
        channels = int.from_bytes(file.read(2), "little")                                           # 22-23  NumOfChannels
        assert channels==2, "channels must be 2"
        sample_rate = int.from_bytes(file.read(4), "little")                                        # 24-27  SampleRate
        ByteRate = int.from_bytes(file.read(4), "little")                                           # 28-31  ByteRate
        BlockAlign = int.from_bytes(file.read(2), "little")                                         # 32-33  BlockAlign
        BitsPerSample = int.from_bytes(file.read(2), "little")                                      # 34-35  BitsPerSample
        assert str(file.read(4), encoding='utf-8')=="data", "\"data\" header incorrect"             # 36-39  data
        datasize = int.from_bytes(file.read(4), "little")                                           # 40-43  datasize

        bytes_per_sample = BitsPerSample//8
        assert ByteRate == channels*sample_rate*bytes_per_sample, "Incorrect ByteRate"
        assert BlockAlign == channels*bytes_per_sample, "Incorrect BlockAlign"
        assert chunksize == datasize + 36, "Incorrect chunksize or datasize"
        
        data = []
        for i in range(datasize//frame_size):
            data.append(file.read(frame_size))
        
        file.close()
        print("Read success")
        
        return data, sample_rate, channels, bytes_per_sample
    except Exception as e:
        print("Error during read")
        print(e)
        return 1


def frames_to_waveform(frames, bytes_per_sample=2, channels=2):
    waveform = []
    if channels == 2:
        for frame in frames:
            for i in range(0, len(frame), channels*bytes_per_sample):
                num1 = struct.unpack('<h', frame[i:i+bytes_per_sample])[0]
                num2 = struct.unpack('<h', frame[i+bytes_per_sample:i+2*bytes_per_sample])[0]
                waveform.append((num1+num2)/32768.0/2.0)    # resize to [-1, 1], average 2 channels
    elif channels == 1:
        for frame in frames:
            for i in range(0, len(frame), channels*bytes_per_sample):
                num1 = struct.unpack('<h', frame[i:i+bytes_per_sample])[0]
                waveform.append(num1/32768.0)
    else:
        raise("channels error")
            
    waveform = numpy.array(waveform)
    return waveform


def waveform_to_frames(waveform, frame_size=4096, bytes_per_sample=2, channels=2, sample_rate=44100):
    assert channels == 2, "Only support 2 channels now"
    assert bytes_per_sample== 2, "Only support 2 byte samples now"
    frames = []
    
    if channels == 2:
        current_size = 0
        frame = b''
        for float in waveform:
            b = struct.pack('h', round(float*32768))
            frame += b  # connect bytes
            frame += b  # 2 channels
            current_size += 1
            
            if current_size == frame_size:    # finished 1 frame
                current_size = 0
                frames.append(frame)
                frame = b''
    elif channels == 1:
        current_size = 0
        frame = b''
        for float in waveform:
            b = struct.pack('h', round(float*32768))
            frame += b  # connect bytes
            current_size += 1
            
            if current_size == frame_size:    # finished 1 frame
                current_size = 0
                frames.append(frame)
                frame = b''
    return frames

#frame total length is doubled due to double channels. len(frames) is also doubled because every frame is broken into 2
def change_frame_size_and_channels(frames, framesize):
    print(len(frames[0]))
    print(framesize)
    assert len(frames[0])//framesize==4, "frame != 4*framesize, unexpected usage"
    newframes=[]
    for frame in frames:
        newframes.append(frame[0:framesize])
        newframes.append(frame[framesize:framesize*2])
        newframes.append(frame[framesize*2:framesize*3])
        newframes.append(frame[framesize*3:framesize*4])
    return newframes