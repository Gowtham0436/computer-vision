// NPZ File Parser for JavaScript
// NPZ files are ZIP archives containing .npy (NumPy array) files

class NPZParser {
    constructor() {
        this.data = null;
    }
    
    /**
     * Parse NPZ file from ArrayBuffer
     * @param {ArrayBuffer} arrayBuffer - Raw NPZ file data
     * @returns {Promise<Object>} Parsed data with arrays
     */
    async parseNPZ(arrayBuffer) {
        try {
            // Load ZIP archive
            const zip = await JSZip.loadAsync(arrayBuffer);
            const result = {};
            
            // Parse each .npy file in the archive
            for (const filename in zip.files) {
                if (filename.endsWith('.npy')) {
                    const file = zip.files[filename];
                    const arrayBuffer = await file.async('arraybuffer');
                    const arrayName = filename.replace('.npy', '');
                    result[arrayName] = this.parseNPY(arrayBuffer);
                }
            }
            
            return result;
        } catch (error) {
            console.error('Error parsing NPZ file:', error);
            throw error;
        }
    }
    
    /**
     * Parse .npy (NumPy array) file
     * @param {ArrayBuffer} arrayBuffer - Raw .npy file data
     * @returns {Object} Parsed array with data and shape
     */
    parseNPY(arrayBuffer) {
        const view = new DataView(arrayBuffer);
        const uint8View = new Uint8Array(arrayBuffer);
        let offset = 0;
        
        // Check magic number (first 6 bytes: 0x93 'N' 'U' 'M' 'P' 'Y')
        const expectedMagic = [0x93, 0x4E, 0x55, 0x4D, 0x50, 0x59];
        let isValidMagic = true;
        
        for (let i = 0; i < 6; i++) {
            if (uint8View[i] !== expectedMagic[i]) {
                isValidMagic = false;
                break;
            }
        }
        
        if (!isValidMagic) {
            const actualMagic = Array.from(uint8View.slice(0, 6));
            console.log('Expected magic:', expectedMagic.map(b => '0x' + b.toString(16)).join(' '));
            console.log('Actual magic:', actualMagic.map(b => '0x' + b.toString(16)).join(' '));
            
            if (uint8View[0] === 0x93 && uint8View[1] === 0x4E) {
                console.warn('Magic number partially matches, continuing...');
            } else {
                throw new Error('Invalid .npy file format - magic number mismatch. Expected: 0x93 0x4E 0x55 0x4D 0x50 0x59');
            }
        }
        offset = 6;
        
        // Read version
        const version = view.getUint8(offset);
        offset += 1;
        
        console.log('NPY version:', version);
        
        // Read header length
        let headerLength;
        let headerStartOffset = offset + 2;
        let headerContentStart = -1;
        
        if (version === 1) {
            const byte0 = uint8View[offset];
            const byte1 = uint8View[offset + 1];
            headerLength = byte0 | (byte1 << 8);
            
            if (headerLength > 1000 || headerLength < 10) {
                const searchStart = offset + 2;
                const searchEnd = Math.min(searchStart + 100, arrayBuffer.byteLength);
                
                for (let i = searchStart; i < searchEnd; i++) {
                    if (uint8View[i] === 0x7B) {
                        headerContentStart = i;
                        break;
                    }
                }
                
                if (headerContentStart > 0) {
                    let headerEnd = searchEnd;
                    let braceCount = 0;
                    
                    for (let i = headerContentStart; i < Math.min(headerContentStart + 500, arrayBuffer.byteLength); i++) {
                        const byte = uint8View[i];
                        if (byte === 0x7B) {
                            braceCount++;
                        } else if (byte === 0x7D) {
                            braceCount--;
                            if (braceCount === 0) {
                                headerEnd = i + 1;
                                break;
                            }
                        }
                    }
                    
                    const actualHeaderContentLength = headerEnd - headerContentStart;
                    const padding = (16 - (actualHeaderContentLength % 16)) % 16;
                    headerLength = actualHeaderContentLength + padding;
                    headerStartOffset = headerContentStart;
                } else {
                    const swappedLength = byte1 | (byte0 << 8);
                    if (swappedLength < 1000 && swappedLength > 10) {
                        headerLength = swappedLength;
                    } else {
                        throw new Error(`Invalid header length: ${headerLength}. Could not find header start.`);
                    }
                }
            }
            offset += 2;
        } else if (version === 2) {
            headerLength = view.getUint32(offset, true);
            offset += 4;
        } else {
            throw new Error(`Unsupported NPY version: ${version}`);
        }
        
        const headerReadOffset = headerContentStart > 0 ? headerContentStart : offset;
        
        if (headerReadOffset + headerLength > arrayBuffer.byteLength) {
            const availableBytes = arrayBuffer.byteLength - headerReadOffset;
            console.warn(`Header length (${headerLength}) exceeds available bytes (${availableBytes}). Reading what we can.`);
            headerLength = availableBytes;
        }
        
        const headerBytes = uint8View.slice(headerReadOffset, headerReadOffset + headerLength);
        
        let headerStr = '';
        for (let i = 0; i < headerBytes.length; i++) {
            const byte = headerBytes[i];
            if (byte === 0) break;
            headerStr += String.fromCharCode(byte);
        }
        
        headerStr = headerStr.trim();
        
        let actualHeaderEnd = headerStr.length;
        for (let i = headerStr.length - 1; i >= 0; i--) {
            const char = headerStr[i];
            if (char !== ' ' && char !== '\n' && char !== '\r' && char !== '\t') {
                actualHeaderEnd = i + 1;
                break;
            }
        }
        const actualHeaderStr = headerStr.substring(0, actualHeaderEnd);
        
        if (headerContentStart > 0) {
            offset = headerContentStart + headerLength;
        } else {
            offset += headerLength;
        }
        
        const header = this.parseHeader(actualHeaderStr);
        
        const dataLength = arrayBuffer.byteLength - offset;
        
        if (dataLength < 0) {
            throw new Error(`Invalid data length: ${dataLength}. Header length may be incorrect.`);
        }
        
        if (dataLength === 0) {
            throw new Error('No data found after header. File may be corrupted.');
        }
        
        const data = new Uint8Array(arrayBuffer, offset, dataLength);
        const typedArray = this.convertToTypedArray(data, header.descr, header.shape);
        
        return {
            data: typedArray,
            shape: header.shape,
            dtype: header.descr,
            fortran_order: header.fortran_order
        };
    }
    
    parseHeader(headerStr) {
        const header = {};
        
        let shapeMatch = headerStr.match(/shape\s*:\s*\(([^)]+)\)/);
        
        if (!shapeMatch) {
            shapeMatch = headerStr.match(/shape\s*:\s*([^\s,}]+(?:,\s*[^\s,}]+)*)/);
        }
        
        if (shapeMatch) {
            const shapeStr = shapeMatch[1].trim();
            if (shapeStr && shapeStr !== '') {
                header.shape = shapeStr.split(',').map(s => {
                    const trimmed = s.trim();
                    const parsed = parseInt(trimmed);
                    if (isNaN(parsed)) {
                        console.warn('Could not parse shape dimension:', trimmed);
                        return null;
                    }
                    return parsed;
                }).filter(n => n !== null);
            } else {
                header.shape = [];
            }
        } else {
            const shapePattern = /\((\d+(?:\s*,\s*\d+)*)\)/;
            const fallbackMatch = headerStr.match(shapePattern);
            if (fallbackMatch) {
                const shapeStr = fallbackMatch[1];
                header.shape = shapeStr.split(',').map(s => parseInt(s.trim())).filter(n => !isNaN(n));
            } else {
                console.warn('Could not parse shape from header:', headerStr.substring(0, 200));
                header.shape = [];
            }
        }
        
        let dtypeMatch = headerStr.match(/'descr'\s*:\s*'([^']+)'/);
        
        if (!dtypeMatch) {
            dtypeMatch = headerStr.match(/"descr"\s*:\s*"([^"]+)"/);
        }
        
        if (!dtypeMatch) {
            dtypeMatch = headerStr.match(/descr\s*:\s*['"]([^'"]+)['"]/);
        }
        
        if (!dtypeMatch) {
            dtypeMatch = headerStr.match(/descr\s*:\s*([^\s,}]+)/);
        }
        
        if (!dtypeMatch) {
            const descrIndex = headerStr.indexOf("'descr'");
            if (descrIndex >= 0) {
                const afterDescr = headerStr.substring(descrIndex + 7);
                dtypeMatch = afterDescr.match(/\s*:\s*'([^']+)'/);
            }
        }
        
        if (!dtypeMatch) {
            const descrIndex = headerStr.indexOf('descr');
            if (descrIndex >= 0) {
                const afterDescr = headerStr.substring(descrIndex);
                dtypeMatch = afterDescr.match(/:\s*['"]([^'"]+)['"]/);
            }
        }
        
        if (dtypeMatch) {
            header.descr = dtypeMatch[1].trim();
            if (header.descr[0] === '|') {
                header.descr = '<' + header.descr.slice(1);
            }
        } else {
            console.warn('Could not parse dtype from header:', headerStr.substring(0, 100));
            header.descr = '<f4';
        }
        
        const fortranMatch = headerStr.match(/fortran_order\s*:\s*(True|False)/);
        header.fortran_order = fortranMatch && fortranMatch[1] === 'True';
        
        return header;
    }
    
    convertToTypedArray(data, dtype, shape) {
        const totalElements = shape.reduce((a, b) => a * b, 1);
        
        let endian = 'little';
        let typeChar = dtype[0];
        let bytes = 1;
        
        if (dtype.length > 1 && (dtype[0] === '<' || dtype[0] === '>')) {
            endian = dtype[0] === '<' ? 'little' : 'big';
            typeChar = dtype[1];
            bytes = parseInt(dtype.slice(2)) || 1;
        } else if (dtype.length > 1 && dtype[0] === 'u') {
            typeChar = 'u';
            bytes = parseInt(dtype.slice(1)) || 1;
        } else if (dtype.length > 1 && dtype[0] === 'i') {
            typeChar = 'i';
            bytes = parseInt(dtype.slice(1)) || 1;
        } else if (dtype.length > 1 && dtype[0] === 'f') {
            typeChar = 'f';
            bytes = parseInt(dtype.slice(1)) || 4;
        }
        
        let TypedArray;
        switch (typeChar) {
            case 'f':
                TypedArray = bytes === 4 ? Float32Array : Float64Array;
                break;
            case 'i':
                TypedArray = bytes === 1 ? Int8Array : 
                            bytes === 2 ? Int16Array : 
                            bytes === 4 ? Int32Array : Int32Array;
                break;
            case 'u':
                TypedArray = bytes === 1 ? Uint8Array : 
                            bytes === 2 ? Uint16Array : 
                            bytes === 4 ? Uint32Array : Uint32Array;
                break;
            default:
                console.warn('Unknown dtype:', dtype, '- defaulting to Uint8Array');
                TypedArray = Uint8Array;
                bytes = 1;
        }
        
        const expectedBytes = totalElements * bytes;
        const availableBytes = data.byteLength;
        
        if (expectedBytes > availableBytes) {
            console.warn(`Expected ${expectedBytes} bytes but got ${availableBytes}. Using available bytes.`);
        }
        
        const bytesToRead = Math.min(expectedBytes, availableBytes);
        const alignedBytes = Math.floor(bytesToRead / bytes) * bytes;
        
        if (alignedBytes === 0) {
            throw new Error(`No data available. Expected ${expectedBytes} bytes but got ${availableBytes}`);
        }
        
        const buffer = data.buffer.slice(data.byteOffset, data.byteOffset + alignedBytes);
        const typedArray = new TypedArray(buffer);
        
        return typedArray;
    }
    
    numpyToMat(npArray) {
        const shape = npArray.shape;
        
        if (shape.length === 2) {
            const rows = shape[0];
            const cols = shape[1];
            const mat = new cv.Mat(rows, cols, cv.CV_8UC1);
            
            const data = npArray.data;
            for (let i = 0; i < rows; i++) {
                for (let j = 0; j < cols; j++) {
                    const idx = i * cols + j;
                    mat.data[idx] = data[idx] > 0 ? 255 : 0;
                }
            }
            
            return mat;
        } else if (shape.length === 3) {
            const numMasks = shape[0];
            const rows = shape[1];
            const cols = shape[2];
            const masks = [];
            
            const data = npArray.data;
            for (let n = 0; n < numMasks; n++) {
                const mat = new cv.Mat(rows, cols, cv.CV_8UC1);
                for (let i = 0; i < rows; i++) {
                    for (let j = 0; j < cols; j++) {
                        const idx = n * rows * cols + i * cols + j;
                        mat.data[i * cols + j] = data[idx] > 0 ? 255 : 0;
                    }
                }
                masks.push(mat);
            }
            
            return masks;
        }
        
        throw new Error('Unsupported array shape');
    }
}

