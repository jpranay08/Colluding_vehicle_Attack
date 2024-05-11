samplepeerrepobsm={'peerrepobsm0': {7: (203.93005986222713, 10.5)}, 'peerrepobsm1': {2: (45.89747569265017, 10.5), 3: (124.88443808675879, 10.5), 4: (50.841587528859364, 0.0), 5: (128.8548232776068, 0.0)}, 'peerrepobsm2': {1: (91.43585502669829, 0.0), 4: (50.841587528859364, 0.0), 6: (5.419763063982572, 10.5)}, 'peerrepobsm3': {1: (91.43585502669829, 0.0), 5: (128.8548232776068, 0.0)}, 'peerrepobsm4': {1: (91.43585502669829, 0.0), 2: (45.89747569265017, 10.5)}, 'peerrepobsm5': {1: (91.43585502669829, 0.0), 3: (124.88443808675879, 10.5)}, 'peerrepobsm6': {2: (45.89747569265017, 10.5)}, 'peerrepobsm7': {0: (211.11003644474044, 0.0)}}

sampleslefrepocollectbsm={'selfrepobsmcollect0': {7: (203.93005986222713, 10.5)}, 'selfrepobsmcollect1': {2: (45.89747569265017, 10.5), 3: (124.88443808675879, 10.5), 4: (50.841587528859364, 0.0), 5: (128.8548232776068, 0.0)}, 'selfrepobsmcollect2': {1: (91.43585502669829, 0.0), 4: (50.841587528859364, 0.0), 6: (5.419763063982572, 10.5)}, 'selfrepobsmcollect3': {1: (91.43585502669829, 0.0), 5: (128.8548232776068, 0.0)}, 'selfrepobsmcollect4': {1: (91.43585502669829, 0.0), 2: (45.89747569265017, 10.5)}, 'selfrepobsmcollect5': {1: (91.43585502669829, 0.0), 3: (124.88443808675879, 10.5)}, 'selfrepobsmcollect6': {2: (45.89747569265017, 10.5)}, 'selfrepobsmcollect7': {0: (211.11003644474044, 0.0)}}

def calculateDecentralizedReputation(peersense:dict,selfrepocollec:dict):
    reputation={}
    for i in peersense.keys():
        vehicleno=i[-1]
        repstr="reputationby"+vehicleno
        reputation[repstr]={}

    for i in peersense.keys():
        selfstr="selfrepobsmcollect"
        vehicleno=i[-1]
        selfstr=selfstr+vehicleno
        reps="reputationby"+vehicleno
        repk=reputation[reps]
        for j in peersense[i].keys():
            print("vehicle list for ",peersense[i],j)
            print("Self repo collection lsit",selfrepocollec[selfstr])
            sk=selfrepocollec[selfstr]
            k=peersense[i]
            print("internal",k[j],j)
            print("self internal",sk[j],j)
            # repk[j]=(repk[j]+ptrust)/2
            
            if(k[j]==sk[j]):
                ptrust=1
                print("the self and peer sense of vehicle",j,"is self:",sk[j],"peer sensed:",k[j])
            else:
                ptrust=-1
            if(j in repk.keys()):
                prepov=repk[j]
                repk[j]=(prepov+ptrust)/2
            else:
                repk[j]=ptrust

        # print("individual peersense",i)
        # print("individual vehicle list",peersense[i])
        # print("self repo bsm collection",selfrepocollec[selfstr])
        # print("reputation by vehicles",reputation)
    

calculateDecentralizedReputation(samplepeerrepobsm,sampleslefrepocollectbsm)