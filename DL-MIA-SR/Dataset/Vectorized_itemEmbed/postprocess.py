import numpy as np


def postprocess(dataProcessed, dataToProcess):
    fw = open(dataProcessed, 'ab')
    with open(dataToProcess, 'r') as fr:
        for line in fr.readlines():
            line = line.strip().split('\t')
            itemID = line[0]
            vector = eval(line[1])
            vector = np.array(list(vector.values()))[None]
            np.savetxt(fw, vector, fmt="%.6f", delimiter=' ', newline='\n')


if __name__ == '__main__':
    dataToProcess = 'beauty_itemMatrix_pre'
    dataProcessed = 'beauty_itemMatrix.txt'
    postprocess(dataProcessed, dataToProcess)

    # dataToProcess = 'ml-1m_itemMatrix_pre'
    # dataProcessed = 'ml-1m_itemMatrix.txt'
    # postprocess(dataProcessed, dataToProcess)


