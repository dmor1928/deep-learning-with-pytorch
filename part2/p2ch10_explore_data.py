from dsets import getCandidateInfoList, getCt, LunaDataset
candidateInfo_list = getCandidateInfoList(requireOnDisk_bool=False)
positiveInfo_list = [x for x in candidateInfo_list if x[0]]
diameter_list = [x[1] for x in positiveInfo_list]

for i in range(0, len(diameter_list), 100):
    print('{:4} {:4.1f} mm'.format(i, diameter_list[i]))

from vis import findPositiveSamples, showCandidate
positiveSample_list = findPositiveSamples()

series_uid = positiveInfo_list[11][2]
showCandidate(series_uid)
