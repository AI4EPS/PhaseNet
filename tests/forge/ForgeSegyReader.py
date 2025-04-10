# Copyright 2025 Motion signal Technologies
# 
# Redistribution and use in source and binary forms, with or without modification, are permitted provided that the following conditions are met:
# 
# 1. Redistributions of source code must retain the above copyright notice, this list of conditions and the following disclaimer.
# 
# 2. Redistributions in binary form must reproduce the above copyright notice, this list of conditions and the following disclaimer in the documentation and/or other materials provided with the distribution.
# 
# 3. Neither the name of the copyright holder nor the names of its contributors may be used to endorse or promote products derived from this software without specific prior written permission.
# 
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

import os
import io
import struct
import datetime
from obspy.io.segy.segy import SEGYFile,SEGYTrace,SEGYTraceHeader,SEGYTraceHeaderTooSmallError
from obspy.io.segy.header import DATA_SAMPLE_FORMAT_SAMPLE_SIZE,DATA_SAMPLE_FORMAT_UNPACK_FUNCTIONS

from obspy.core import AttribDict
from obspy import Stream, Trace, UTCDateTime

class FORGE_SEGYTrace(SEGYTrace):
    '''
    Custom traces structure for FORGE segy files
    '''
    
    def _read_trace(self, 
                    force_trace_length=None,
                    unpack_headers=False, 
                    headonly=False):
        """
        Reads the complete next header starting at the file pointer at
        self.file.

        :type unpack_headers: bool
        :param unpack_headers: Determines whether or not all headers will be
            unpacked during reading the file. Has a huge impact on the memory
            usage and the performance. They can be unpacked on-the-fly after
            being read. Defaults to False.
        :type headonly: bool
        :param headonly: Determines whether or not the actual data records
            will be read and unpacked. Has a huge impact on memory usage. Data
            will not be unpackable on-the-fly after reading the file.
            Defaults to False.
        """
        
        trace_header = self.file.read(240)

        # Check if it is smaller than 240 byte.
        if len(trace_header) != 240:
            msg = 'The trace header needs to be 240 bytes long'
            raise SEGYTraceHeaderTooSmallError(msg)
        self.header = SEGYTraceHeader(trace_header,
                                      endian=self.endian,
                                      unpack_headers=unpack_headers)

        # Overirde the number of samples bytes 173-176    
        numberOfSamples = struct.unpack('i',trace_header[172:176])[0]
        self.header.number_of_samples_in_this_trace=numberOfSamples

        # Millisecond part of time stamp bytes 169-170  
        self.millisecond_of_second= struct.unpack('h',trace_header[168:170])[0]
        #print("milliSec=",milliSec)
    
        # Micro-second part of time stamp bytes 171-172   
        self.microsecond_of_millisecond=struct.unpack('h',trace_header[170:172])[0]
        
        # The number of samples in the current trace.
        npts = self.header.number_of_samples_in_this_trace
        self.npts = npts
        
        # Do a sanity check if there is enough data left.
        pos = self.file.tell()
        data_left = self.filesize - pos
        data_needed = DATA_SAMPLE_FORMAT_SAMPLE_SIZE[self.data_encoding] * \
            npts
        if npts < 1 or data_needed > data_left:
            msg = """
                  Too little data left in the file to unpack it according to
                  its trace header. This is most likely either due to a wrong
                  byte order or a corrupt file.
                  """.strip()
            raise SEGYTraceReadingError(msg)
        
        if headonly:
            # skip reading the data, but still advance the file
            self.file.seek(data_needed, 1)
            # build a function for reading data from the disk on the fly
            self.unpack_data = OnTheFlyDataUnpacker(
                DATA_SAMPLE_FORMAT_UNPACK_FUNCTIONS[self.data_encoding],
                self.file.name, self.file.mode, pos, npts, endian=self.endian)
        else:
            # Unpack the data.
            self.data = DATA_SAMPLE_FORMAT_UNPACK_FUNCTIONS[
                self.data_encoding](self.file, npts, endian=self.endian)

    def to_obspy_trace(self, overide_sample_rate=None, unpack_trace_headers=False, headonly=False):

        tr=super(FORGE_SEGYTrace,self).to_obspy_trace(unpack_trace_headers=False, headonly=False)

        # Fix the sample rate
        if overide_sample_rate!=None: 
            tr.stats.sampling_rate=overide_sample_rate
            tr.stats.delta=1./overide_sample_rate

        tr.stats.starttime += datetime.timedelta(milliseconds=self.millisecond_of_second,
                                                microseconds=self.microsecond_of_millisecond)

        
        return tr

        

    def write(self, file, data_encoding=None, endian=None):
        raise Exception("Writing traces not supported..")

class FORGE_SEGYFile(SEGYFile):
    '''
    Custom reader for Forge segy Files
    '''
        
    
    def _read_traces(self, unpack_headers=False, headonly=False,
                     yield_each_trace=False):
        """
        Reads the actual traces starting at the current file pointer position
        to the end of the file.

        :type unpack_headers: bool
        :param unpack_headers: Determines whether or not all headers will be
            unpacked during reading the file. Has a huge impact on the memory
            usage and the performance. They can be unpacked on-the-fly after
            being read. Defaults to False.

        :type headonly: bool
        :param headonly: Determines whether or not the actual data records
            will be read and unpacked. Has a huge impact on memory usage. Data
            will not be unpackable on-the-fly after reading the file.
            Defaults to False.

        :type yield_each_trace: bool
        :param yield_each_trace: If True, it will yield each trace after it
            has been read. This enables a simple implementation of a
            streaming interface to read SEG-Y files. Read traces will no
            longer be collected in ``self.traces`` list if this is set to
            ``True``.
        """
        
        self.traces = []
        # Determine the filesize once.
        if isinstance(self.file, io.BytesIO):
            pos = self.file.tell()
            self.file.seek(0, 2)  # go t end of file
            filesize = self.file.tell()
            self.file.seek(pos, 0)
        else:
            filesize = os.fstat(self.file.fileno())[6]


        # Big loop to read all data traces.
        while True:
            # Read and as soon as the trace header is too small abort.
            try:
                trace = FORGE_SEGYTrace(self.file, self.data_encoding, self.endian,
                                  unpack_headers=unpack_headers,
                                  filesize=filesize, headonly=headonly)

                if yield_each_trace:
                    yield trace
                else:
                    self.traces.append(trace)
            except SEGYTraceHeaderTooSmallError:
                break

        return []
    
    def _write_textual_header(self,**kwargs):
        raise Exception("write functionality not supported")

    def _write(self,**kwargs):
        raise Exception("write functionality not supported")
    
    def write(self,**kwargs):
        raise Exception("write functionality not supported")
    
def read(filename, headonly=False, unpack_trace_headers=False,**kwargs):
    '''
    Reads a Forge SEGY file to produce and ObsPy data stream basically 
     a slight tweak on obspy.io.core._read_segy
    '''

    # Read the file into SEGY representation
    with open(filename,'rb') as fd:
        reader=FORGE_SEGYFile(file=fd,
                             headonly=headonly,

                             **kwargs
                             )
    # Create the stream object.
    stream = Stream()
    # SEGY has several file headers that apply to all traces. They will be
    # stored in Stream.stats.
    stream.stats = AttribDict()
    # Get the textual file header.
    textual_file_header = reader.textual_file_header
    # The binary file header will be a new AttribDict
    binary_file_header = AttribDict()
    for key, value in reader.binary_file_header.__dict__.items():
        setattr(binary_file_header, key, value)
    # Get the data encoding and the endianness from the first trace.
    data_encoding = reader.traces[0].data_encoding
    endian = reader.traces[0].endian
    textual_file_header_encoding = reader.textual_header_encoding.upper()
    # Add the file wide headers.
    stream.stats.textual_file_header = textual_file_header
    stream.stats.binary_file_header = binary_file_header
    # Also set the data encoding, endianness and the encoding of the
    # textual_file_header.
    stream.stats.data_encoding = data_encoding
    stream.stats.endian = endian
    stream.stats.textual_file_header_encoding = \
        textual_file_header_encoding

    # Convert traces to ObsPy Trace objects.
    for tr in reader.traces:
        stream.append(tr.to_obspy_trace(
            overide_sample_rate=1./(1.e-6*reader.binary_file_header.sample_interval_in_microseconds),
            headonly=headonly,
            unpack_trace_headers=unpack_trace_headers))


    
    return stream