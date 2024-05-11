import itertools
import time
def safe_divide(numerator, denominator):
    if (denominator < 0) | (denominator == 0):
        return numerator
    else:
        return (numerator / denominator)


# default prior values are uniform uninformative
def massComb(masses, prior0=0.5, prior1=0.5, prior01=0):

    # the space in the hypothesis space that is not in the evidence space
    prior_theta = 1 - (prior0 + prior1 + prior01)

    # since sum of m(A) must equal 1 there may be frame of descernment is
    # what's left over
    for i in range(len(masses)):
        masses[i].append( 1 - sum(masses[i]))
    

    ##########################
    # PERFORM MASS COMBINATION
    #########################
    #print("The different mass functions are:")
    intrsxn_array_dim = len(masses[1])
    # set the dimenensions of the mass comb. matrix.
    intrsxn_array = [
        [0 for j in range(intrsxn_array_dim)] for i in range(intrsxn_array_dim)]

    ############### BEGIN: Combining all bpa's ##############################
    for i in range(0,len(masses)-1):
        if i == 0:
            K = 1  
            m0 = masses[0][0]
            m1 = masses[0][1]
            m01 = masses[0][2]
            m_theta = masses[0][3]
            #print("First mass assignment. m0:", m0, "m1: ", m1,
            #      "m01: ", m01, "m_theta: ", m_theta, "K: ", K)

        new_mass = [[m0, m1, m01, m_theta]]
    
        for col in range(intrsxn_array_dim):
            for row in range(intrsxn_array_dim):
                intrsxn_array[row][col] = round(new_mass[0][col]*masses[i + 1][row],2)
        #print(new_mass,"new mass")
        

        # CALCULATE K - the measure of conflict
        K = intrsxn_array[0][1] + intrsxn_array[1][0]
        if(K==1.0):
            K=0.9
        # Calculate belief functions
        m0 = (intrsxn_array[0][0] + intrsxn_array[2][0] + intrsxn_array[3][0]
            + intrsxn_array[0][2] + intrsxn_array[0][3]) / (1 - K)
        m1 = (intrsxn_array[1][1] + intrsxn_array[1][2] + intrsxn_array[1][3]
            + intrsxn_array[2][1] + intrsxn_array[3][1]) / (1 - K)
        m01 = intrsxn_array[2][3] / (1 - K)
        # normalize to emphasise agreement
        m_theta = intrsxn_array[3][3] / (1 - K)
        #print("Next mass assignment. m0:", m0, "m1: ", m1,
        #      "m01: ", m01, "m_theta: ", m_theta, "K: ", K)
    ############### END: Combining all bpa's ###############################

    #print("m0:", m0, "m1: ", m1, "m01: ",
    #      m01, "m_theta: ", m_theta, "K: ", K)
    # INCLUDE PRIOR INFORMATION
    #print("\n")
    #print("prior0: ", prior0, "prior1: ", prior1,
    #      "prior01: ", prior01, "prior_theta: ", prior_theta)
    # basic certainty assignment (bca) and normalize
    certainty_denominator = (safe_divide(numerator = m0, denominator = prior0) + safe_divide(numerator = m1, denominator = prior1)
                             + 
                             safe_divide(
                                 numerator=m01, denominator=prior01)
                             + safe_divide(numerator=m_theta, denominator=prior_theta))
    # print(certainty_denominator)
    # time.sleep(2)
    if(certainty_denominator==0):
        certainty_denominator=0.5
    C0 = safe_divide(numerator=m0, denominator=prior0) / \
        certainty_denominator
    C1 = safe_divide(numerator=m1, denominator=prior1) / \
        certainty_denominator
    C01 = safe_divide(
        numerator=m01, denominator=prior01) / certainty_denominator
    C_theta = safe_divide(
        numerator=m_theta, denominator=prior_theta) / certainty_denominator
    #print("C0: ", C0, "C1: ", C1, "C01: ", C01, "C_theta: ", C_theta)
    #print("C0 + C1 + C01 + C_theta: ", C0 + C1 + C01 + C_theta, "\n")

    #print("inrsxn_array:", intrsxn_array[0][1])
    
    blf0 = round(m0,2)
    blf1 = round(m1,2)
    blf01 = round(m0 + m1 + m01,2)

    plsb0 = round(m0 + m01 + m_theta,2)
    plsb1 = round(m1 + m01 + m_theta,2)
    plsb_theta = 1

    mass_fxn_values = {"blf0": blf0, "blf1": blf1, "blf01": blf01, \
                        "plsb0": plsb0, "plsb1": plsb1, "plsb_theta": plsb_theta}

    return mass_fxn_values


masses0=[0.1,0.9,0]
masses1=[0.1,0.9,0]
masses2=[0.1,0.9,0]
m3=[0.9,0.1,0]

#print(massComb(masses=[masses0,masses1,masses2,m3]))

def calculateMassFunction(mass_fxn_values,adjacent_list,reputation_list):
    plsb0=mass_fxn_values["plsb0"]
    plsb1=mass_fxn_values["plsb1"]
    masses={}
    for i in range(len(adjacent_list)):
        mass=[]
        currentveh=adjacent_list[i]
        currentvehreport=reputation_list[i]
        diff=abs(plsb1-currentvehreport)
        if(diff==0):
            mass.append(0.1)
            mass.append(0.9)
            mass.append(0)
        else:
            mass.append(diff)
            mass.append(1-diff)
            mass.append(0)
        masses[currentveh]=mass
    # print("massesss",masses)
    return masses


# def getReportOfvehicles():

def calculateRepGyawali(pRep,cTrust,smoothing_fac=0.2):
    return round(((smoothing_fac*(pRep))+((1-smoothing_fac)*cTrust)),2)





def calculateCentralizedReputation(vehicleList, selfReport,centralizedRSUReputation):
    
    for i in vehicleList.keys():
        if(vehicleList[i]==selfReport[i][0]):
            cTrust=0.9
        else:
            cTrust=0.1
        currVehicleRep=centralizedRSUReputation.get(i,0)
        centralizedRSUReputation[i]=calculateRepGyawali(currVehicleRep,cTrust,0.2)
    # print("the repitation is ",centralizedRSUReputation)
    return centralizedRSUReputation


def calculateVehilceReputationReport(centralizedRSUReputation, reputationByVehicle,repDiffByRSU):
    vehRepStr="reputationby"
    for i in reputationByVehicle.keys():
        totaldiff=0
        
        for j in reputationByVehicle[i]:
            # print("??             ??? \n")
            
            repDiff=reputationByVehicle[i][j]-(centralizedRSUReputation[j])

            # print("the diference between the reputaiton is ",repDiff)
            # print("??             ??? \n")
            #  added the buffer of 0.4 because of the different update policies
            if(abs(repDiff)>0.3):
                totaldiff=totaldiff+abs(repDiff)
        arr=repDiffByRSU.get(i,[])
        arr.append(totaldiff)
        repDiffByRSU[i]=arr
    # print("reputation difference by rsu",repDiffByRSU)
    return repDiffByRSU
 

# def calculateVehRepu(repDiffByRSU,ReputationByRsuToVehReport):
#     for i in repDiffByRSU.keys():
#         print(repDiffByRSU)
#         if(all(earlier >= later for earlier, later in zip(repDiffByRSU[i], repDiffByRSU[i][1:]))):
#             rsuRepToVehReport=0.9
#         else:
#             rsuRepToVehReport=0.1
#         pRep=ReputationByRsuToVehReport.get(i,0)
#         pRep=calculateRepGyawali(pRep,rsuRepToVehReport,0)
#         ReputationByRsuToVehReport[i]=pRep
#     repDiffByRSU={}


#     print("\n \n \n \n \n the final reputation of the vehicle by its reports \n \n \n \n \n ",ReputationByRsuToVehReport, "the reputation difference repDiffByRSU",repDiffByRSU)

#     return ReputationByRsuToVehReport
    

