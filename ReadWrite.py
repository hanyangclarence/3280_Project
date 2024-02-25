import struct

# write the contents of data to filepath
# returns 0 on success, return 1 and print error message if failed
# data is frames[]
def write_wav(data, filepath, rate=44100, channels=2, bytes_per_sample=2):
    if not data:
        print("No data to write")
        return
    try:
        file = open(filepath, "wb")
        
        datasize = len(data) * len(data[0])   #every frame element seems to have length 4096
        
        file.write("RIFF".encode())                     #0-3    RIFF
        file.write(struct.pack('i', 36+datasize))       #4-7    chunksize = datasize + 36
        file.write("WAVEfmt ".encode())                 #8-15   WAVEfmt(SPACE)
        file.write(struct.pack('i', 16))                #16-19  SubchunkSize = 16
        file.write(struct.pack('h', 1))                 #20-21  AudioFormat = 1
        file.write(struct.pack('h', channels))          #22-23  NumOfChannels
        file.write(struct.pack('i', rate))              #24-27  SampleRate
        byte_rate = rate * channels * bytes_per_sample
        file.write(struct.pack('i', byte_rate))         #28-31  ByteRate
        block_align = channels * bytes_per_sample
        file.write(struct.pack('h', block_align))       #32-33  BlockAlign
        bits_per_sample = bytes_per_sample * 8
        file.write(struct.pack('h', bits_per_sample))   #34-35  BitsPerSample
        file.write("data".encode())                     #36-39  data
        file.write(struct.pack('i', datasize))          #40-43  datasize
        for d in data:
            file.write(d)           
        
        file.close()
        print("Write success")
        return 0
    except Exception as e:
        print("Error during write")
        print(e)
        return 1
    
# reads wav from filepath, and returns a tuple (frames[], sample_rate, channels, bytes_per_sample)
# returns 1 and prints error message if failed
# remember to check the return value!
def read_wav(filepath, frame_size=4096):
    try:
        file = open(filepath, "rb")
        
        assert str(file.read(4), encoding='utf-8')=="RIFF", "File does not start with RIFF"         #0-3    RIFF
        chunksize = int.from_bytes(file.read(4), "little")                                          #4-7    chunksize = datasize + 36
        assert str(file.read(8), encoding='utf-8')=="WAVEfmt ", "File has incorrect WAVEfmt part"   #8-15   WAVEfmt(SPACE)
        assert int.from_bytes(file.read(4), "little")==16, "SubchunkSize is not 16"                 #16-19  SubchunkSize = 16
        assert int.from_bytes(file.read(2), "little")==1, "AudioFormat is not 1"                    #20-21  AudioFormat = 1
        channels = int.from_bytes(file.read(2), "little")                                           #22-23  NumOfChannels
        sample_rate = int.from_bytes(file.read(4), "little")                                        #24-27  SampleRate
        ByteRate = int.from_bytes(file.read(4), "little")                                           #28-31  ByteRate
        BlockAlign = int.from_bytes(file.read(2), "little")                                         #32-33  BlockAlign
        BitsPerSample = int.from_bytes(file.read(2), "little")                                      #34-35  BitsPerSample
        assert str(file.read(4), encoding='utf-8')=="data", "\"data\" header incorrect"             #36-39  data
        datasize = int.from_bytes(file.read(4), "little")                                           #40-43  datasize
        
        
        bytes_per_sample = BitsPerSample//8
        assert ByteRate == channels*sample_rate*bytes_per_sample, "Incorrect ByteRate"
        assert BlockAlign == channels*bytes_per_sample, "Incorrect BlockAlign"
        assert chunksize == datasize + 36, "Incorrect chunksize or datasize"
        
        data = []
        for i in range(datasize//frame_size):
            data.append(file.read(frame_size))
        
        file.close()
        print("Read success")
        print("samplerate is "+str(sample_rate))
        
        data = break_into_frames(join_frames(data))
        return (data, sample_rate, channels, bytes_per_sample)
    except Exception as e:
        print("Error during read")
        print(e)
        return 1
    
def join_frames(frames):
    return b''.join(frames)

def break_into_frames(bytedata, frame_size=4096):
    frames = []
    for i in range(0, len(bytedata), frame_size):
        frames.append(bytedata[i:i+frame_size])
    return frames