import sys
sys.path.append(".qlearning")
sys.path.append("tmp/ppo\\actor_torch_ppo")
import numpy as np
import time
import itertools
import pandas as pd
import random
import logging
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
import math as ma
from random import choice
from collections import defaultdict
from typing import List, Tuple, Dict, TYPE_CHECKING, Optional
from math import radians, cos, sin, asin, sqrt
from dempster import massComb, calculateMassFunction,calculateCentralizedReputation,calculateVehilceReputationReport

import gym
from qlearning import Agent
#from actorCritic import Agent
# from ppo import Agent
from utilsCE import plot_learning_curve
from customEnv import CustomEnv


from highway_env.road.lane import LineType, StraightLane, AbstractLane
from highway_env.road.objects import Landmark

if TYPE_CHECKING:
    from highway_env.vehicle import kinematics
    from highway_env.road import objects

logger = logging.getLogger(__name__)






#     for i in range(n_games):
#         score=0
#         done= False
#         observation= env.reset()
#         while not done:
#             action=agent.choose_action(observation)
#             observation_, reward, done , info =env.step(action,)
#             score+=reward
#             agent.store_transition(observation, action, reward, observation_, done )
#             agent.learn()
#             observation= observation_
#         scores.append(score)
#         eps_history.append(agent.epsilon)
#         avg_score =np.mean(scores[-100:])
#         print('episode', i, 'score %.2f' % score,
#               'average score %.2f' % avg_score,
#               'epsilon %.2f' % agent.epsilon)
#     x=[i+1 for i in range(n_games)]
#     filename= 'CustomEnv.png'
#     plot_learning_curve(x, scores, eps_history, filename)

#dempster code test
# masses0=[0.1,0.9,0]
# masses1=[0.1,0.9,0]
# masses2=[0.1,0.9,0]
# m3=[0.9,0.1,0]

# print(massComb(masses=[masses0,masses1,masses2,m3]))


#fault generating code

#  for j in i.road.vehicles:
#             noisefa=random.randint(0,98)
#             noisefa=noisefa%(len(i.road.vehicles)-1)
#             noise1=random.randint(0,9)
#             if(j.id==noisefa):
#                 faultVechicleposition[j.id]=(j.position[0]+noise1,j.position[1]+noise1)
#                 # print("ok")
#                 (x,y)=faultVechicleposition[j.id]
#                 # print("x,y",x,y)
#                 faultvehiclex.append(x)
#                 faultvehicley.append(y)
#                 # print("type of x coridb",type(x))
#                 # print("type of faultvechile",type(faultVechicleposition[j.id]))
   
#             else:
#                 faultVechicleposition[j.id]=(j.position[0],j.position[1])
#                 (x,y)=faultVechicleposition[j.id]
#                 faultvehiclex.append(x)
#                 faultvehicley.append(y)
#                 # print("type of x coridb",type(x))
#                 # print("type of faultvechile",type(faultVechicleposition[j.id]))
#             coordinates.append((j.position[0],j.position[1]))
#             vehiclename[j.id]=(j.position[0],j.position[1])
#         #     print("ivehicle",i.position,"each vehicle position",j.position)
#         # print("faultyy vehicle position",faultVechicleposition)
#         plotVehiclePosition(vehiclex,vehicley,faultvehiclex,faultvehicley)
#         for ke in newvehicleposition:
#             arra1=[]
#             arra1.append(vehiclename[ke])
#             for ko in vehiclename:
#                 if(ke!=ko):
#                     arr=newvehicleposition[ke]
#                     arr.append(vehiclename[ko])
#                     arra1.append(faultVechicleposition[ko])
#                     newfaultposition[ke]=arra1
#                     newvehicleposition[ke]=arr


LaneIndex = Tuple[str, str, int]
Route = List[LaneIndex]
peerfaultyreports=[]
reputation={}
smoothingFactor_space=np.round(np.arange(0.6, 0.95, 0.01), 2)
reward={}
rewardIncrements={}
oldvehicleposition={}
reputationincrements={}
plausibilityIncrements={}
repuationStepIncrement={}
centralizedRSUreputation={}
centralizedRSURepIteration={}
massfuncionsFromCentralizedReput={}
massFunctionsFromLevel1Plausibility={}
massFunctionsFromLevel2Plausibility={}
ReputationByRsuToVehReport={}
rewardValues=[]
repDiffByRSU={}
finalMassFunction={}
finalMassFunctionIterations={}
finalMaliciousVehicles=[]
centralizedcalculation={}
stepsize=0
checknewIteration=[]
attackedVehicle=[-1]
rsuActionstart=0
rsuActionDiff=6
actionIterations=[]
episode_count=0
malGyawaliVehicle=set()
env =CustomEnv()
finalPlausibilityAvg=0
N=20
cc=0
score=0
# print("gamma",0.995,"epsilon",1.0,"lr",0.001,"input_dims",[3],"batch_size",64,"n_actions","eps_dec",0.01)
agent= Agent(gamma=0.995, epsilon=1.0, batch_size =64, n_actions=35,eps_end =0.01, input_dims=[3], lr=0.001)
#agent = Agent(alpha=1e-5, n_actions=34)
# agent= Agent(gamma=0.99, epsilon=1.0, batch_size =64, n_actions=35,eps_end =0.01, input_dims=[3], lr=0.003)
# ppo agent = Agent(n_actions=34, batch_size=5, 
                    # alpha=0.0003, n_epochs=4, 
                    # input_dims=env.observation_space.shape)
floder="DQN150"
scores, eps_history =[],[]
action =None
n_steps = 0
score_history = []
avg_score = 0
learn_iters = 0
best_score=0
observation=env.dqreset()
time.sleep(5)
print("the observation is ",observation)
n_games= 500
def rewardReputation(reputation):
    reputationThres=0.7
    for i in reputation.keys():
        if(reputation[i]<reputationThres):
            #print("reward for vechile",i,"is","-1")
            reward[i]=reward[i]-1
        else:
            #print("reward for vechile",i,"is","+1")
            reward[i]=reward[i]+1
       
def validationPoints(newvehiclecoordinate,maliciousV,adjacentV):
    validationPoint={}
    #print("in validation")
    #print(newvehiclecoordinate)
    print("malicious vehicles",maliciousV)
    # print("Adjacent List",adjacentV)
    for i in maliciousV:
        arr1=[]
        if i in adjacentV:
            #print("adjacent of i",adjacentV[i])
            for j in adjacentV[i]:
                noise1=random.randint(1,9)
                #print("noise1 ",noise1)
                sarr=newvehiclecoordinate[j]
                print(sarr)
                (x,y)=sarr
                print((x+noise1,y+noise1))
                arr1.append((x+noise1,y+noise1))
            validationPoint[i]=arr1
    #print("validation points",validationPoint)
    return validationPoint


# rep calcuation froim thesis
 #formula= 1- (1/denom)
 #denom=1+exp(expval)
 #expval= -(rep-D)/sigma
# formulaa= 1-(1/(1+exponential(-(reputation-D)/sigma)))

def getPhiRep(pRep):
    D=200
    sigma=200
    phiRep=1-(1/(1+np.exp(-(pRep-D)/sigma)))
    return phiRep

def getNormalizedRep(pRep):
    D=200
    return (pRep/D)

def calculateRepThesis(pRep,ctrust):
    D=200
    theta=20
    thetaInv=1/theta
    curphiVal=getPhiRep(pRep)
    norPRep=getNormalizedRep(pRep)

    cRep=pRep+thetaInv*curphiVal * D * (ctrust-norPRep)
    return cRep
# The reward is the no of true messages to the total no of messages.

#implementing dempster shafer in calculating repitation.


def calculateVehRepu(repDiffByRSU,ReputationByRsuToVehReport):
    global centralizedRSURepIteration
    for i in repDiffByRSU.keys():

        if(i in centralizedRSURepIteration.keys()):
            arr1=centralizedRSURepIteration.get(i,[])
            # print("the arr of centralized is ",arr1,centralizedRSURepIteration)
           
        else:
            arr1=[]
            # print("the arr1 of centralized is ",arr1,centralizedRSURepItera6tion)
        # print(repDiffByRSU)
        if(all(earlier >= later for earlier, later in zip(repDiffByRSU[i], repDiffByRSU[i][1:])) or (all([ v == 0 for v in repDiffByRSU[i] ])) ):
            rsuRepToVehReport=0.9
        else:
            rsuRepToVehReport=0.1
        pRep=ReputationByRsuToVehReport.get(i,0)
        pRep=calculateRepGyawali(pRep,rsuRepToVehReport,0.1)
        ReputationByRsuToVehReport[i]=pRep
        arr1.append(pRep)
        centralizedRSURepIteration[i]=arr1
    # print("centralized rsu rep Iteration",centralizedRSURepIteration )
    repDiffByRSU={}


    print("\n \n \n \n \n the final reputation of the vehicle by its reports \n \n \n \n \n ",ReputationByRsuToVehReport, "the reputation difference repDiffByRSU",repDiffByRSU)

    return ReputationByRsuToVehReport
def calculateMassFunctionFromCentralizedRpeort(ReputationByRsuToVehReport):
    global massfuncionsFromCentralizedReput
    # print("the mass function of centralized is",massfuncionsFromCentralizedReput)
    for i in ReputationByRsuToVehReport.keys():
        massfunction=[]
        vehid=int(i.split("reputationby")[1])
        massfunction.append(round(1-ReputationByRsuToVehReport[i],2))
        massfunction.append(ReputationByRsuToVehReport[i])
        # print("reputation in centralized ",ReputationByRsuToVehReport[i])
        massfunction.append(0)
        # print("the mass function of the veh ",vehid,"from centralized reputation is ",massfunction)
        # print("the mass function of previous one",massfuncionsFromCentralizedReput.get(vehid))
        massfuncionsFromCentralizedReput[vehid]=massfunction
   
       
    # print("the mass fucntion from the centralized is ",massfuncionsFromCentralizedReput)
    return massfuncionsFromCentralizedReput


def getTheMaliciousMassFunctionFromMassFunction(massFunctionLevel1, massFunctionLevel2,centralizedMassFunction,attackedVehicle,folder):
    global floder
    if(len(attackedVehicle)>2):
        maliciousL1Mass= [v for key in attackedVehicle if key in massFunctionLevel1 for v in massFunctionLevel1[key]]
        maliciousL2Mass= [v for key in attackedVehicle if key in massFunctionLevel2 for v in massFunctionLevel2[key]] 
        maliciouscMass= []
        
        for v in attackedVehicle:
            maliciouscMass.append(centralizedMassFunction[v])
        savePlotVals(maliciousL1Mass,"Centralized reputation","Centralized reputation based on validatation by RSU","maliciouslevel1",folder+"maliciouslevel1" )
        savePlotVals(maliciousL2Mass,"Centralized reputation","Centralized reputation based on validatation by RSU","maliciouslevel2",floder+"maliciouslevel2" )
        savePlotVals(maliciouscMass,"Centralized reputation","Centralized reputation based on validatation by RSU","maliciouscentralized",floder+"maliciouscentralized" )



       
def combineMassFunctionsFromThreeLevels(massFunctionLevel1, massFunctionLevel2,centralizedMassFunction):
    global finalMassFunction
    global finalMassFunctionIterations
    global finalMaliciousVehicles
    N=3
    finalMaliciousVehicles=[]
    for i in centralizedMassFunction.keys():
       
        vehMassCollection=[]
        # print("the centralized reprt",centralizedMassFunction[i])
        vehMassCollection.append(centralizedMassFunction[i][:N])
        level1Mass=massFunctionLevel1.get(int(i),0)
        # print("level 1 mass of veh",i," is ",level1Mass)
        if(level1Mass!=0):
            vehMassCollection.append(level1Mass[:N])
        level2Mass=massFunctionLevel2.get(int(i),0)
        # print("level 2 mass veh",i," is ",level2Mass)
        if(level2Mass!=0):
            vehMassCollection.append(level2Mass[:N])
        prevRep=finalMassFunction.get(i,0)
        if(i in finalMassFunctionIterations.keys()):
            arr1=finalMassFunctionIterations.get(i,[])
            # print("the arr of centralized is ",arr1,centralizedRSURepIteration)
           
        else:
            arr1=[]
        if(len(vehMassCollection)>1):
            # print("the veh mass of veh ",i ,'is', vehMassCollection)
            finalplausi=massComb(vehMassCollection)
            # print("final plausibilty",finalplausi)
            # if(finalplausi["plsb0"]<0):
            #     print("\n the final plausiblity of the vehicle is less than 0",i,finalplausi["plsb0"] ,vehMassCollection,"\n")
            #     time.sleep(3)
            cuurentFinalRepu=calculateRepThesis(prevRep,finalplausi["plsb0"])
            if(finalplausi["plsb0"]>0.2):
                finalMaliciousVehicles.append(i)
            #test code
            # if(finalplausi["plsb0"]>0.5):
            #     cuurentFinalRepu=calculateRepThesis(prevRep,1)
            # elif(finalplausi["plsb0"]>0.3):
            #     cuurentFinalRepu=calculateRepThesis(prevRep,0.5)
            # else:
            #     cuurentFinalRepu=calculateRepThesis(prevRep,0)
            arr1.append(cuurentFinalRepu)
            finalMassFunctionIterations[i]=arr1
            finalMassFunction[int(i)]= cuurentFinalRepu  
        else:
            # print("the vehicle with less no of levels passed is", i)
            cuurentFinalRepu=calculateRepThesis(prevRep,vehMassCollection[0][0])
            arr1.append(cuurentFinalRepu)
            finalMassFunctionIterations[i]=arr1
            finalMassFunction[int(i)]= cuurentFinalRepu
    calculateAvgFinalPlausibility(finalMaliciousVehicles,finalMassFunction)
# [0, 1, 2, 5, 4, 7] 2, 5, 0, 4, 1, 6]
# [3, 2, 4, 7, 6, 0]
    # print("the final valure of mass function is ",finalMassFunction)

# def claculateGyawaliReward(level1MassPlausibilityVals,centralizedMassFunction):
#     global finalGyaMaliciousVehicles
#     for i in centralizedMassFunction.keys():
#         if(finalplausi["plsb0"]>0.3):
#                 finalGyaMaliciousVehicles.append(i)


def calculateAvgFinalPlausibility(finalMaliciousVehicles,finalMassFunction):
    global finalPlausibilityAvg
    sum=0
    if(len(finalMaliciousVehicles)>0):
        for i in finalMaliciousVehicles:
            sum=+finalMassFunction[int(i)]
        finalPlausibilityAvg=sum/len(finalMaliciousVehicles)
    else:
        finalPlausibilityAvg=0
   






def calculateavgReputation(adjacentList, reputation,matrix):
    # global avgReputation
    global stepsize
    global plausibilityIncrements
    global massFunctionsFromLevel1Plausibility
    global massFunctionsFromLevel2Plausibility
    global malGyawaliVehicle
    massFunctionsFromLevel1Plausibility={}
    repStr='reputationby'

   
    externalAvg=0
    finalMasses={}
    for adjKe in adjacentList.keys():
        finalMasses[adjKe]=[]

    # print("matrix", matrix)
    for adjKe in adjacentList.keys():
        x=adjacentList[adjKe]
       
        # print(adjacentList)
        # print(x)
        # print(reputation)
        adjmass=[]
        sum=0
        masses=[]
        reput=[]
        intialMasses=[]
        for i in x:
            mass=[]
           
            repuOfV=reputation[repStr+str(i)]
            # print("the i is",i,"the rep pof v is",repuOfV,"adj",adjKe)
            # print("the reputaion of ",adjKe, "by its one of the adj vehicle", x ,i ,"is",repuOfV[adjKe])
            reput.append(repuOfV.get(adjKe,0))
           
            sum=sum+repuOfV[adjKe]
            mass.append(round(1-repuOfV[adjKe],2))
            mass.append(round(repuOfV[adjKe],2))
            mass.append(0)
            masses.append(mass)
        # print("the reputaion of ",adjKe, "by its adj vehicle",reput)
        # print("masses of vehicle",adjKe, "by its neighbours",masses)
        if(len(x)>1):

            internalAvg=sum/len(x)
            mass_fxn_values=massComb(masses)
            if(mass_fxn_values["plsb1"]<0.9):
                malGyawaliVehicle.add(adjKe)

            # adjmass.append(mass_fxn_values["plsb0"])
            # adjmass.append(mass_fxn_values["plsb1"])
            # adjmass.append(0)
            # currentMasses=finalMasses[adjKe]
            # currentMasses.append(adjmass)
            # print("belief and plausibility values of vehicle",adjKe," is ",mass_fxn_values)
           
            indiMasses=calculateMassFunction(mass_fxn_values,x,reput)
            if(mass_fxn_values["plsb1"]==1):
                intialMasses=[0.05,0.75,0]
            else:
                intialMasses.append(((mass_fxn_values["plsb0"])/2))
                intialMasses.append(((mass_fxn_values["plsb1"])/2))
                intialMasses.append(0)
           
            massFunctionsFromLevel1Plausibility[adjKe]=intialMasses

            for i in indiMasses.keys():
                currentMasses=finalMasses[i]
                currentMasses.append(indiMasses[i])
                finalMasses[i]=currentMasses


        else:
            internalAvg=sum
            finalMasses[i]=masses
            # print("maassses",masses)
            if(masses[0][0]>0.2):
                malGyawaliVehicle.add(adjKe)
            # massN=[0.5,0.5,0]
            massFunctionsFromLevel1Plausibility[adjKe]=[0.5,0.5,0]
            # print("intital maasses is ",masses[0])
            # print("not enough data to implement dempster shafer model as there is only one adjacent vehicle")
        # print("internal Avg of vehicle",adjKe,"is",internalAvg)
        externalAvg+=internalAvg
    # print("The external avg",externalAvg/len(adjacentList))
    # print("the initial round mass functions are \n", massFunctionsFromLevel1Plausibility)
    avgReputation=externalAvg/len(adjacentList)
    # print("the final masses ",finalMasses)
    for i in finalMasses.keys():
        secondMassFunction=[]
        if(i in plausibilityIncrements.keys()):
            arr1=plausibilityIncrements.get(i,[])
        else:
            arr1=[]
        if(len(finalMasses[i])>1):
            fiPlaus=massComb(finalMasses[i])
            # print("the final plausibility of the vehicles",i,fiPlaus)
            secondMassFunction.append(((fiPlaus["plsb0"])/2))
            secondMassFunction.append(((fiPlaus["plsb1"])/2))
            secondMassFunction.append(0)
            arr1.append(fiPlaus["plsb0"])
            plausibilityIncrements[i]=arr1
            massFunctionsFromLevel2Plausibility[i]=secondMassFunction

    # print("Mass functions in the second round ",massFunctionsFromLevel2Plausibility)

    return avgReputation


def calculateReward(adjacentList, reputation):
    # global rewardmalGyawaliVehicle
    # print("length of adjacentlist is",len(adjacentList))
    for adjKe in adjacentList.keys():
        repuOfV=reputation[repStr+str(adjKe)]
        if any([True for k,v in repuOfV.items() if v == -1]):
            falseCount+=1
            # print("the vehicle",adjKe,"contains negative values",repuOfV.values())
    reward=(total-falseCount)/total
    return reward
   


def calculateRepGyawali(pRep,cTrust,smoothing_fac=0.5):
    return round(((smoothing_fac*(pRep))+((1-smoothing_fac)*cTrust)),2)

def addWhiteGuassianNoise(point):
    noise_stddev = 0.9
    noise = np.random.normal(2, noise_stddev, 1)
    # print("\nWhite Guassian Noise",noise)
    # print("Point Before noise",point)
    (x,y)=point
    x=x+noise[0]
    y=y+noise[0]
    # print("Point after noise",(x,y),"\n")
    return (x,y)
   
def validatePoints(validationPoints,vechilecoordinates,maliciousVehicle,Adjacent):
    maliciousValidation={}
    for veh in maliciousVehicle:
        if veh in Adjacent:
           arr=[]
           count=0
           for i in Adjacent[veh]:
               if i not in maliciousVehicle:
                    valid=validationPoints[veh]
                    reported=vechilecoordinates[veh]
                    if(i<veh):
                        peerreport=reported[i+1]
                    else:
                        peerreport=reported[i]
                    #print("peer report",peerreport,"validation point",valid[count])
                    if(peerreport==valid[count]):
                        arr.append("true")
                    else:
                        arr.append("false")
                    count=count+1
           maliciousValidation[veh]=arr
        # print("maliciousValidation",maliciousValidation)

def plotValidationpoints(validationPoints,vechilecoordinates,maliciousVehicle,Adjacent):
    global stepsize
    if(stepsize>200 and stepsize%100==0):
        plt.rcParams["figure.figsize"] = [7, 5]
        plt.rcParams["figure.autolayout"] = True
        plt.axis([200, 400, -5, 30])
        plt.ylabel("Longitude",fontsize=12)
        plt.xlabel("Latitude",fontsize=12)
        if(len(maliciousVehicle)%2!=0):
            plotsize=len(maliciousVehicle)+1
        else:
            plotsize=len(maliciousVehicle)
        plotcol=plotsize/2
        if(plotcol<5):
            plotcol=3
        label=["malicious vehicle reports","self reports of vehicles","RSU Validating Points"]
        count=1
        for veh in maliciousVehicle:            
            if veh in Adjacent:
                a=[]
                b=[]
                x=[]
                y=[]
                for i in Adjacent[veh]:
                   
                    reported=vechilecoordinates[veh]
                    if(i<veh):
                        peerreport=reported[i+1]
                    else:
                        peerreport=reported[i]
                    (a1,b1)=peerreport
                    a.append(a1)
                    b.append(b1)
                    print("VALUES FROM PLOT VALIDATION ")
                    print("PEER REPORT by malicious vehicle for "+str(i)+"by vehicle"+str(veh)+"is",str(a),str(b))
                    selfrepo=vechilecoordinates[i]
                    (x1,y1)=selfrepo[0]
                    x.append(x1)
                    y.append(y1)
                    print("self report of vehicle"+str(i)+"is",str(a),str(b))
                   
                c=[]
                d=[]
                for j in validationPoints[veh]:
                    (c1,d1)=j
                    c.append(c1)
                    d.append(d1)
                    #print("validation points from rrsu to veh",str(veh)+"is",str(c),str(d))
                plt.subplot(2,plotcol,count)
                plt.title("validation of"+str(veh))
                plt.plot(a,b,"r*")              
                plt.plot(x,y,"go")
                plt.plot(c,d,'bv')
                count=count+1
        plt.legend(label)
        plt.show()
                # for k in vechilecoordinates[veh]:
                #     print("""""""""""""""""""""""?????????????????????????????""""",k)
                #     (x,y)=k
               
       


def plotSingleValidationpoints(validationPoints,vechilecoordinates,maliciousVehicle,Adjacent):
    global stepsize
    if(stepsize>200 and stepsize%50==0):
        plt.rcParams["figure.figsize"] = [7, 5]
        plt.rcParams["figure.autolayout"] = True
        plt.ylabel("Longitude",fontsize=12)
        plt.xlabel("Latitude",fontsize=12)
        #plt.axis([200, 450, -5, 20])
        if(len(maliciousVehicle)%2!=0):
            plotsize=len(maliciousVehicle)+1
        else:
            plotsize=len(maliciousVehicle)
        plotcol=plotsize/2
        if(plotcol<5):
            plotcol=3
        label=["Malicious vehicle reports","Self reports of vehicles","RSU validating points"]
        count=1
        for hev in maliciousVehicle:
            if hev in Adjacent:
                veh=hev

        vehlabel=[]
        if veh in Adjacent:
            a=[]
            b=[]
            x=[]
            y=[]
            for i in Adjacent[veh]:
                if i not in maliciousVehicle:
                    vehlabel.append(i)
                    reported=vechilecoordinates[veh]
                    if(i<veh):
                        peerreport=reported[i+1]
                    else:
                        peerreport=reported[i]
                    (a1,b1)=peerreport
                    a.append(a1)
                    b.append(b1)
                    #print("VALUES FROM PLOT VALIDATION ")
                    #print("PEER REPORT by malicious vehicle for "+str(i)+"by vehicle"+str(veh)+"is",str(a),str(b))
                    selfrepo=vechilecoordinates[i]
                    (x1,y1)=selfrepo[0]
                    x.append(x1)
                    y.append(y1)
                    print("self report of vehicle"+str(i)+"is",str(a),str(b))
               
            c=[]
            d=[]
            for j in validationPoints[veh]:
                (c1,d1)=j
                c.append(c1)
                d.append(d1)
                print("validation points from rrsu to veh",str(veh)+"is",str(c),str(d))
            plt.title("Validation of "+str(veh))
           
            plt.plot(a,b,"r*")
            print("lenght of malicious",len(a))              
            plt.plot(x,y,"go")
            print("lenght of self",len(x))              

            plt.plot(c,d,'bv')
            print("lenght of self",len(c))              

            for c2,d2,vp in zip(c,d,vehlabel):
                plt.text(c2+10, d2, '{} '.format(vp),fontsize=12)
            count=count+1
            plt.legend(label,fontsize=12)
            plt.tight_layout()
            plt.xticks(fontsize=12)
            plt.yticks(fontsize=12)
            plt.show()
               


def haversine(lat1, lon1, lat2, lon2):

      R = 500 #3959.87433 # this is in miles.  For Earth radius in kilometers use 3959.87433 km

      dLat = radians(lat2 - lat1)
      dLon = radians(lon2 - lon1)
      lat1 = radians(lat1)
      lat2 = radians(lat2)

      a = sin(dLat/2)**2 + cos(lat1)*cos(lat2)*sin(dLon/2)**2
      c = 2*asin(sqrt(a))

      return R * c

def pythagorus(lat1,lon1,lat2,lon2):
    R = 500  # radius of the earth in km
    x = (radians(lon2) - radians(lon1)) * cos(0.5 * (radians(lat2) + radians(lat1)))
    y = radians(lat2) - radians(lat1)
    d = R * sqrt(x*x + y*y)
    return d

def plotReputation():
    global reputationincrements
    markers=["o","v","^","<",">","1","2","3","4","8","s","p","P","*","+","x","X","D","d"]
    count=1
    # if(len(reputationincrements)%2!=0):
    #     plotsize=len(reputationincrements)+1
    # else:
    #     plotsize=len(reputationincrements)
    # plotcol=plotsize/2
   
    # plt.rcParams["figure.figsize"] = [13.50, 8.50]
    # plt.rcParams["figure.autolayout"] = True


    fig, ax = plt.subplots(figsize=(7, 5))
    plt.ylabel("Reputation",fontsize=12)
    plt.xlabel("No of Steps",fontsize=12)
    plt.title("Reputation plot")
    l= []
    for i in reputationincrements:
        #print("value of i",i)
        if(np.any(reputationincrements[i])):
            print("value of i",i)
            # plt.rcParams["figure.figsize"] = [7.50, 3.50]
            # plt.rcParams["figure.autolayout"] = True

            x = reputationincrements[i]
            y = repuationStepIncrement[i]
            l.append(i)
           
            ax.plot(y, x)
            # ax.text(x[-1]*1.05, y[-1], 'vehicle'+str(i), color=plt.gca().get_lines()[0].get_color())
            count=count+1
    # for line, name in zip(ax.lines, l):
    #             x = line.get_xdata()[-1]
    #             ax.annotate(name, xy=(x,1), xytext=(6,0), color=line.get_color(),
    #                         xycoords = ax.get_xaxis_transform(), textcoords="offset points",
    #                         size=5, va="bottom")            
    ax.legend(l)
    plt.xticks(fontsize=12)
    plt.yticks(fontsize=12)
    plt.show()  


import os
my_path = os.path.abspath(__file__)
def plotPlausibility(plausibility,ylabel,title,name):
    global episode_count
   
    # Naming the x-axis, y-axis and the whole graph
    plt.xlabel("steps ")
    plt.ylabel(ylabel)
    plt.title(title)
   

    for i in plausibility.keys():
        # Plotting both the curves simultaneously
       
        arr=plausibility[i]
        X = np.arange(0, len(arr), 1)
        plt.plot(X,arr , label=i)
        # Adding legend, which helps us recognize the curve according to it's color
    plt.legend()
   
    # To load the display window

    plt.savefig(name+"/"+name+ str(episode_count)+".png")
    # plt.close()




def savePlotVals(plausibility,ylabel,title,foldername,name):
    global episode_count
    with open(str(foldername)+'/'+str(name)+".txt", 'a') as f:
        f.write(foldername+str(episode_count)+' = ' + repr(plausibility) + '\n')

def plotList(list,ylabel,title, name):
   
   
    # Naming the x-axis, y-axis and the whole graph
    plt.xlabel("Steps")
    plt.ylabel(ylabel)
    plt.title(title)
   

   
        # Plotting both the curves simultaneously
       
    X = np.arange(0, len(list), 1)
    plt.plot(X,list )
        # Adding legend, which helps us recognize the curve according to it's color
    plt.legend()
   
    # To load the display window
    plt.savefig(name+"/"+name+ str(episode_count)+".png")
    # plt.close()

def resetReputation():
    global reputation
    global reputationincrements
    global repuationStepIncrement
    global plausibilityIncrements
    global reward
    global rewardValues
    global score
    global done
    global observation
    global cc
    global centralizedRSUreputation
    global ReputationByRsuToVehReport
    global repDiffByRSU
    global centralizedRSURepIteration
    global massfuncionsFromCentralizedReput
    global finalMassFunction
    global finalMassFunctionIterations
    global centralizedcalculation
    global actionIterations
    global malGyawaliVehicle
    global score_history
    global avg_score
    global best_score
    global agent
    # print("inside reset",reputation)
    reputation={}
    reward={}
    # print("plausibilty of 0",plausibilityIncrements)
    plausibilityIncrements={}
    reputationincrements={}
    repuationStepIncrement={}
    centralizedRSUreputation={}
    ReputationByRsuToVehReport={}
    centralizedRSURepIteration={}
    finalMassFunctionIterations={}
    centralizedcalculation={}
    actionIterations=[]
    repDiffByRSU={}
    falseMsgFromMaliciousVehicle=0
    massfuncionsFromCentralizedReput={}
    rewardValues=[]
    cc=0
    finalMassFunction={}
    done= False
    observation= env.dqreset()
    incrementEpisodeCount()
    malGyawaliVehicle=set()
    # ppo
    # score_history.append(score)
    # avg_score = np.mean(score_history[-100:])

    # if avg_score > best_score:
    #     best_score = avg_score
    #     agent.save_models()
    # score=0
def incrementStepSize()->None:
    global rsuActionstart
    global stepsize
    global rsuActionDiff
    if(stepsize>19 and stepsize%20==0):
        rsuActionstart=stepsize
        rsuActionDiff=stepsize-rsuActionstart
    else:
         rsuActionDiff=stepsize-rsuActionstart
    stepsize=stepsize+1


def incrementEpisodeCount() -> None:
    global episode_count
    episode_count=episode_count+1
   
   

# def getRSUAction()->None:
#     global rsuActionstart

# def calculatefaultReputation(vehiclename:dict , faultvehicle:dict,matrix,*args,**kwargs)-> None:
#     global stepsize
#     print("first matrix",matrix)
#     if len(reputation)==0:
#         for x in vehiclename.keys():
#             reputation[x]=0
#     reputationmatrix=np.copy(matrix)
#     for i in range(len(matrix)):
#         for j in range(len(matrix)):
#             if(matrix[i][j]==1):
#                 arrself=vehiclename[j]
#                 print("arrself",arrself)

#                 valself=arrself[0]
#                 arrrepot=faultvehicle[i]
#                 print("arrrepot",arrrepot)
#                 if(i<j):
#                     valrepot=arrrepot[j]
#                 else:
#                     valrepot=arrrepot[j+1]
#                 if(valself==valrepot):
#                     reputationmatrix[i][j]=1
#                 else:
#                     reputationmatrix[i][j]=-1
#     print("matrix",matrix)
#     for i in range(len(matrix)):
#         n=0
#         if(i in reputation.keys()):
#             sum=reputation[i]
           
#             arr=reputationincrements.get(i,[])
#         else:
#             sum=0
#             arr=[]
#             steparr=[]
#             steparr.append(stepsize)
#             arr.append(sum)
#             reputationincrements[i]=arr
#             repuationStepIncrement[i]=steparr
#         for j in range(len(matrix)):
#             if(reputationmatrix[j][i]!=0):
#                 n=n+1
#             sum=sum+reputationmatrix[j][i]
           
#         if(n>0):
#             n=n+1
#             avg=sum/n
#         else:
#             avg=sum
#         reputation[i]=avg
#         arr.append(avg)
#         steparr=repuationStepIncrement.get(i,[])
#         steparr.append(stepsize)
#         reputationincrements[i]=arr
#         repuationStepIncrement[i]=steparr
#     print("reputaiyon",reputation)
#     print("reputation",reputationincrements)


def calculateReputation(vehiclename:dict,matrix,*args,**kwargs)-> None:
    global stepsize
    print("first matrix",matrix)
    if len(reputation)==0:
        for x in vehiclename.keys():
            reputation[x]=0
            reward[x]=0
    reputationmatrix=np.copy(matrix)
    for i in range(len(matrix)):
        for j in range(len(matrix)):
            if(matrix[i][j]==1):
                peerarr=vehiclename[i]

                if(i<j):
                    peerrepo=peerarr[j]
                else:
                    peerrepo=peerarr[j+1]
                #print("peerrepo of vehicle",j,"by vehicle",i,peerrepo)
                selfarr=vehiclename[j]
                selfrepo=selfarr[0]

                #print("selfrepo of vehicle",j,selfrepo)
                # valself=arrself[0]
                # print("arrrepot",arrrepot)
                # if(i<j):
                #     valrepot=arrrepot[j]
                # else:
                #     valrepot=arrrepot[j+1]
                if(peerrepo==selfrepo and peerrepo!=-1):
                    reputationmatrix[i][j]=1
                else:
                    reputationmatrix[i][j]=-1
    # print("matrix",matrix)
    # print("reputationmatrix",reputationmatrix)
    for i in range(len(matrix)):
        n=0
        if(i in reputation.keys()):
            sum=0
            prev=reputation[i]
            arr=reputationincrements.get(i,[])
        else:
            sum=0
            arr=[]
            steparr=[]
            steparr.append(stepsize)
            arr.append(sum)
            prev=0
            # reputationincrements[i]=arr
            # repuationStepIncrement[i]=steparr
        for j in range(len(matrix)):
            if(reputationmatrix[i][j]!=0):
                n=n+1
            sum=sum+reputationmatrix[i][j]
           
        if(n>0):
            avg=sum/n
        else:
            avg=sum
        reputation[i]=(avg+prev)/2
        arr.append(avg)
        steparr=repuationStepIncrement.get(i,[])
        steparr.append(stepsize)
        reputationincrements[i]=arr
        repuationStepIncrement[i]=steparr
    rewardReputation(reputation)
    malicious=maliciousVehicles(reputation)
   
    # print("Avg reputation",reputation)
    # print("reputation",reputationincrements)
    return malicious

def maliciousVehicles(reputation):
    maliciousV=[]
    for i in reputation.keys():
        if (reputation[i]<0.5):
            maliciousV.append(i)
    # print(maliciousV)
    return maliciousV




def plotVehiclePosition(vehiclenamex,vehiclenamey,faultvehiclex, faultvehicley)-> None:
    global stepsize
    print("stepsize",stepsize)
    label=["Normal vehicle reports","Malicious vehicle reports"]
    if(stepsize>200 and stepsize%100==0):
        plt.rcParams["figure.figsize"] = [7, 5]
        plt.rcParams["figure.autolayout"] = True
        print("len of fault0",len(faultvehicley),faultvehicley)
        x = vehiclenamex
        y = vehiclenamey
        z = faultvehiclex
        a = faultvehicley


        plt.plot(x, y, 'b*')
        plt.axis([0, 500, -5, 20])

        plt.plot(z, a, 'rv')
        # plt.text(20,100,"red is faultvehicle coordinates")
        # plt.text(0,100,"blue is vehicle coordinates")
        flag=0
        for i, j in zip(x, y):
            flag+=1
            plt.text(i, j+0.5, '({} )'.format(flag))
        flag=0
        for i, j in zip(z, a):
            flag+=1
            plt.text(i,j+5.5, '({})'.format(flag))
        plt.legend(label)
        plt.show()




def plotMalNorVehiclePosition(vechileposition:dict,maliciousVechile,Adjacent)-> None:
    global stepsize
    print("stepsize",stepsize)
    vellabel=["Normal vehicle self report","Normal vehicle peer report","Malicious vehicle self report","Malicious vechile peer report"]
    if(stepsize>100 and stepsize%100==0):
        mal=1
        if(len(maliciousVechile)>0):
            mal=maliciousVechile[1]
        for i1 in vechileposition:
            #print("inside for in plot")
            if i1 not in maliciousVechile:
                nor=i1
        plt.rcParams["figure.figsize"] = [7, 5]
        plt.rcParams["figure.autolayout"] = True
        plt.axis([200, 500, -5, 20])
        plt.ylabel("Longitude",fontsize=12)
        plt.xlabel("Latitude",fontsize=12)
        plt.title("Self and peer reports of vehicles")
        flag=0
        number = len(vechileposition)
        cmap = plt.get_cmap('gnuplot')
        colors = [cmap(i) for i in np.linspace(0, 1, number)]
        for k in range(2):
            if k==0:
                i=nor
                for j in range(len(vechileposition[i])):
                    sarr=vechileposition[i]
                    if(j==0):
                        (x,y)=sarr[j]
                        plt.plot(x, y, 'g*')
                        plt.text(x, y+0.5, '{}'.format(i),fontsize=12)
                    else:
                        if(sarr[j]!=-1):
                            c="C"+str(i)
                            (x,y)=sarr[j]
                            plt.plot(x, y, 'bv')
                            plt.text(x-10, y-2.5, '{},{}'.format(j,i),fontsize=12)
                   
            else:
                i=mal
                for j in range(len(vechileposition[i])):
                    sarr=vechileposition[i]
                    #print("ploting the mal")
                    if(j==0):
                        (x,y)=sarr[j]
                        plt.plot(x, y, 'g*')
                        plt.text(x, y+0.5, '{}'.format(i),fontsize=12)
                    else:
                        if(sarr[j]!=-1):
                            c="C"+str(i)
                            (x,y)=sarr[j]
                            plt.plot(x, y, 'r^')
                            plt.text(x, y+2, '{},{}'.format(j,i),fontsize=12)
        for m in range(2):
            for n in Adjacent[mal]:
                saar=vechileposition[n]
                (x,y)=saar[0]
                plt.plot(x,y,'g>')
                plt.text(x,y+0.5,'{}'.format(n),fontsize=12)
            for n1 in Adjacent[nor]:
                saar=vechileposition[n1]
                (x,y)=saar[0]
                plt.plot(x,y,'g>')
                plt.text(x,y+0.5,'{}'.format(n1),fontsize=12)
           
                        #
        #plt.legend( vellabel)  
        plt.xticks(fontsize=12)
        plt.yticks(fontsize=12)
        plt.show()
        plotMalNorVehiclePositionwithoutpoints(vechileposition,mal,nor,Adjacent)
def plotMalNorVehiclePositionwithoutpoints(vechileposition:dict,mall:int,norm1:int,Adjacent:dict)-> None:
    global stepsize
    print("stepsize",stepsize)
    vellabel=["Normal vehicle self report","Normal vehicle peer report","Malicious vehicle self report","Malicious vechile peer report"]
   
    mal=mall
    nor=norm1
    plt.rcParams["figure.figsize"] = [7, 5]
    plt.rcParams["figure.autolayout"] = True
    plt.axis([200, 500, -5, 20])
    plt.ylabel("Longitude",fontsize=12)
    plt.xlabel("Latitude",fontsize=12)
    flag=0
    number = len(vechileposition)
    cmap = plt.get_cmap('gnuplot')
    colors = [cmap(i) for i in np.linspace(0, 1, number)]
    for k in range(2):
        if k==0:
            i=norm1
            for j in range(len(vechileposition[i])):
                sarr=vechileposition[i]
                if(j==0):
                    (x,y)=sarr[j]
                    plt.plot(x, y, 'g*')
                   
                else:
                    if(sarr[j]!=-1):
                        c="C"+str(i)
                        (x,y)=sarr[j]
                        plt.plot(x, y, 'bv')
        else:
            i=mall
            for j in range(len(vechileposition[i])):
                sarr=vechileposition[i]
                #print("ploting the mal")
                if(j==0):
                    (x,y)=sarr[j]
                    plt.plot(x, y, 'g*')
                else:
                    if(sarr[j]!=-1):
                        c="C"+str(i)
                        (x,y)=sarr[j]
                        plt.plot(x, y, 'r^')
    for m in range(2):
        for n in Adjacent[mal]:
            saar=vechileposition[n]
            (x,y)=saar[0]
            plt.plot(x,y,'g>')
        for n1 in Adjacent[nor]:
            saar=vechileposition[n1]
            (x,y)=saar[0]
            plt.plot(x,y,'g>')
    plt.xticks(fontsize=12)
    plt.yticks(fontsize=12)
    plt.show()

# def calculateDecentralizedReputation(slefRepoBsm:dict,peerRepoBsm:dict,adjacentcent):
def calculateDecentralizedReputation(peersense:dict,selfrepocollec:dict, smootingFactor, matrix):
    global reputation
    splitstr="peerrepobsm"
    for i in peersense.keys():
        selfstr="selfrepobsmcollect"
       
        vehicleno=i.split("peerrepobsm")[1]
        selfstr=selfstr+vehicleno
        reps="reputationby"+vehicleno
        if(reps in reputation.keys()):
            repk=reputation[reps]
        else:
            reputation[reps]={}
            repk=reputation[reps]
        for j in peersense[i].keys():
            # print("vehicle list for ",peersense[i],j)
            # print("Self repo collection lsit",selfrepocollec[selfstr])
            sk=selfrepocollec[selfstr]
           
            k=peersense[i]
            # print("internal",k[j],j)
            # print("self internal",sk[j],j)
            # repk[j]=(repk[j]+ptrust)/2
           
            if(k[j]==sk[j]):
                ptrust=0.9
                # print("the self and peer sense of vehicle",j,"by vehicle ",i,"is self:",sk[j],"peer sensed:",k[j])
            else:
                ptrust=0.1
                # print("the self and peer report of malicious vehicle is ",j,"by vehicle ",i,"its self location is :",sk[j], " peer sensed  ",k[j])
            if(j in repk.keys() and repk[j]!={}):
                prepov=repk[j]
                repk[j]=calculateRepGyawali(prepov,ptrust,0.9)# calculateRepThesis(prepov,ptrust)

                #calculateRepGyawali
            else:
                #time.sleep(10)
                repk[j]=ptrust
    print("reeputation of vehicles in the way",reputation)
    return reputation


def plotVehiclePosition(vechileposition:dict)-> None:
    global stepsize
    print("stepsize",stepsize)
    if(stepsize>100 and stepsize%120==0):
        plt.rcParams["figure.figsize"] = [7, 5]
        plt.rcParams["figure.autolayout"] = True
        plt.axis([200, 500, -5, 20])
        flag=0
        number = len(vechileposition)
        cmap = plt.get_cmap('gnuplot')
        colors = [cmap(i) for i in np.linspace(0, 1, number)]
        for i in vechileposition:
            flag=+1
            for j in range(len(vechileposition[i])):
                sarr=vechileposition[i]
                if(j==0):
                    (x,y)=sarr[j]
                    plt.plot(x, y, 'b*')
                    plt.text(x, y+0.5, '{}'.format(i),fontsize=12)
                else:
                    if(sarr[j]!=-1):
                        c="C"+str(i)
                        (x,y)=sarr[j]
                        plt.plot(x, y, 'r*')
                        #plt.text(x, y+0.5+i, '{},{}'.format(j,i))
        plt.xticks(fontsize=12)
        plt.yticks(fontsize=12)
        plt.show()
   

class RoadNetwork(object):
   
    graph: Dict[str, Dict[str, List[AbstractLane]]]

    def __init__(self):
        self.graph = {}

    def add_lane(self, _from: str, _to: str, lane: AbstractLane) -> None:
        """
        A lane is encoded as an edge in the road network.

        :param _from: the node at which the lane starts.
        :param _to: the node at which the lane ends.
        :param AbstractLane lane: the lane geometry.
        """
        if _from not in self.graph:
            self.graph[_from] = {}
        if _to not in self.graph[_from]:
            self.graph[_from][_to] = []
        self.graph[_from][_to].append(lane)

    def get_lane(self, index: LaneIndex) -> AbstractLane:
        """
        Get the lane geometry corresponding to a given index in the road network.

        :param index: a tuple (origin node, destination node, lane id on the road).
        :return: the corresponding lane geometry.
        """
        _from, _to, _id = index
        if _id is None and len(self.graph[_from][_to]) == 1:
            _id = 0
        return self.graph[_from][_to][_id]

    def get_closest_lane_index(self, position: np.ndarray, heading: Optional[float] = None) -> LaneIndex:
        """
        Get the index of the lane closest to a world position.

        :param position: a world position [m].
        :param heading: a heading angle [rad].
        :return: the index of the closest lane.
        """
        indexes, distances = [], []
        for _from, to_dict in self.graph.items():
            for _to, lanes in to_dict.items():
                for _id, l in enumerate(lanes):
                    distances.append(l.distance_with_heading(position, heading))
                    indexes.append((_from, _to, _id))
        return indexes[int(np.argmin(distances))]

    def next_lane(self, current_index: LaneIndex, route: Route = None, position: np.ndarray = None,
                  np_random: np.random.RandomState = np.random) -> LaneIndex:
        """
        Get the index of the next lane that should be followed after finishing the current lane.

        - If a plan is available and matches with current lane, follow it.
        - Else, pick next road randomly.
        - If it has the same number of lanes as current road, stay in the same lane.
        - Else, pick next road's closest lane.
        :param current_index: the index of the current lane.
        :param route: the planned route, if any.
        :param position: the vehicle position.
        :param np_random: a source of randomness.
        :return: the index of the next lane to be followed when current lane is finished.
        """
        _from, _to, _id = current_index
        next_to = None
        # Pick next road according to planned route
        if route:
            if route[0][:2] == current_index[:2]:  # We just finished the first step of the route, drop it.
                route.pop(0)
            if route and route[0][0] == _to:  # Next road in route is starting at the end of current road.
                _, next_to, _ = route[0]
            elif route:
                logger.warning("Route {} does not start after current road {}.".format(route[0], current_index))
        # Randomly pick next road
        if not next_to:
            try:
                next_to = list(self.graph[_to].keys())[np_random.randint(len(self.graph[_to]))]
            except KeyError:
                # logger.warning("End of lane reached.")
                return current_index

        # If next road has same number of lane, stay on the same lane
        if len(self.graph[_from][_to]) == len(self.graph[_to][next_to]):
            next_id = _id
        # Else, pick closest lane
        else:
            lanes = range(len(self.graph[_to][next_to]))
            next_id = min(lanes,
                          key=lambda l: self.get_lane((_to, next_to, l)).distance(position))

        return _to, next_to, next_id

    def bfs_paths(self, start: str, goal: str) -> List[List[str]]:
        """
        Breadth-first search of all routes from start to goal.

        :param start: starting node
        :param goal: goal node
        :return: list of paths from start to goal.
        """
        queue = [(start, [start])]
        while queue:
            (node, path) = queue.pop(0)
            if node not in self.graph:
                yield []
            for _next in set(self.graph[node].keys()) - set(path):
                if _next == goal:
                    yield path + [_next]
                elif _next in self.graph:
                    queue.append((_next, path + [_next]))

    def shortest_path(self, start: str, goal: str) -> List[str]:
        """
        Breadth-first search of shortest path from start to goal.

        :param start: starting node
        :param goal: goal node
        :return: shortest path from start to goal.
        """
        return next(self.bfs_paths(start, goal), [])

    def all_side_lanes(self, lane_index: LaneIndex) -> List[LaneIndex]:
        """
        :param lane_index: the index of a lane.
        :return: all lanes belonging to the same road.
        """
        return [(lane_index[0], lane_index[1], i) for i in range(len(self.graph[lane_index[0]][lane_index[1]]))]

    def side_lanes(self, lane_index: LaneIndex) -> List[LaneIndex]:
        """
                :param lane_index: the index of a lane.
                :return: indexes of lanes next to a an input lane, to its right or left.
                """
        _from, _to, _id = lane_index
        lanes = []
        if _id > 0:
            lanes.append((_from, _to, _id - 1))
        if _id < len(self.graph[_from][_to]) - 1:
            lanes.append((_from, _to, _id + 1))
        return lanes

    @staticmethod
    def is_same_road(lane_index_1: LaneIndex, lane_index_2: LaneIndex, same_lane: bool = False) -> bool:
        """Is lane 1 in the same road as lane 2?"""
        return lane_index_1[:2] == lane_index_2[:2] and (not same_lane or lane_index_1[2] == lane_index_2[2])

    @staticmethod
    def is_leading_to_road(lane_index_1: LaneIndex, lane_index_2: LaneIndex, same_lane: bool = False) -> bool:
        """Is lane 1 leading to of lane 2?"""
        return lane_index_1[1] == lane_index_2[0] and (not same_lane or lane_index_1[2] == lane_index_2[2])

    def is_connected_road(self, lane_index_1: LaneIndex, lane_index_2: LaneIndex, route: Route = None,
                          same_lane: bool = False, depth: int = 0) -> bool:
        """
        Is the lane 2 leading to a road within lane 1's route?

        Vehicles on these lanes must be considered for collisions.
        :param lane_index_1: origin lane
        :param lane_index_2: target lane
        :param route: route from origin lane, if any
        :param same_lane: compare lane id
        :param depth: search depth from lane 1 along its route
        :return: whether the roads are connected
        """
        if RoadNetwork.is_same_road(lane_index_2, lane_index_1, same_lane) \
                or RoadNetwork.is_leading_to_road(lane_index_2, lane_index_1, same_lane):
            return True
        if depth > 0:
            if route and route[0][:2] == lane_index_1[:2]:
                # Route is starting at current road, skip it
                return self.is_connected_road(lane_index_1, lane_index_2, route[1:], same_lane, depth)
            elif route and route[0][0] == lane_index_1[1]:
                # Route is continuing from current road, follow it
                return self.is_connected_road(route[0], lane_index_2, route[1:], same_lane, depth - 1)
            else:
                # Recursively search all roads at intersection
                _from, _to, _id = lane_index_1
                return any([self.is_connected_road((_to, l1_to, _id), lane_index_2, route, same_lane, depth - 1)
                            for l1_to in self.graph.get(_to, {}).keys()])
        return False

    def lanes_list(self) -> List[AbstractLane]:
        return [lane for to in self.graph.values() for ids in to.values() for lane in ids]

    @staticmethod
    def straight_road_network(lanes: int = 4, length: float = 10000, angle: float = 0) -> 'RoadNetwork':
        net = RoadNetwork()
        for lane in range(lanes):
            origin = np.array([0, lane * StraightLane.DEFAULT_WIDTH])
            end = np.array([length, lane * StraightLane.DEFAULT_WIDTH])
            rotation = np.array([[np.cos(angle), np.sin(angle)], [-np.sin(angle), np.cos(angle)]])
            origin = rotation @ origin
            end = rotation @ end
            line_types = [LineType.CONTINUOUS_LINE if lane == 0 else LineType.STRIPED,
                          LineType.CONTINUOUS_LINE if lane == lanes - 1 else LineType.NONE]
            net.add_lane("0", "1", StraightLane(origin, end, line_types=line_types))
        return net

    def position_heading_along_route(self, route: Route, longitudinal: float, lateral: float) \
            -> Tuple[np.ndarray, float]:
        """
        Get the absolute position and heading along a route composed of several lanes at some local coordinates.

        :param route: a planned route, list of lane indexes
        :param longitudinal: longitudinal position
        :param lateral: : lateral position
        :return: position, heading
        """
        while len(route) > 1 and longitudinal > self.get_lane(route[0]).length:
            longitudinal -= self.get_lane(route[0]).length
            route = route[1:]
        return self.get_lane(route[0]).position(longitudinal, lateral), self.get_lane(route[0]).heading_at(longitudinal)


class Road(object):
    """A road is a set of lanes, and a set of vehicles driving on these lanes."""

    def __init__(self,
                 network: RoadNetwork = None,
                 vehicles: List['kinematics.Vehicle'] = None,
                 road_objects: List['objects.RoadObject'] = None,
                 np_random: np.random.RandomState = None,
                 record_history: bool = False) -> None:
        """
        New road.

        :param network: the road network describing the lanes
        :param vehicles: the vehicles driving on the road
        :param road_objects: the objects on the road including obstacles and landmarks
        :param np.random.RandomState np_random: a random number generator for vehicle behaviour
        :param record_history: whether the recent trajectories of vehicles should be recorded for display
        """
        self.network = network
        self.vehicles = vehicles or []
        self.objects = road_objects or []
        self.np_random = np_random if np_random else np.random.RandomState()
        self.record_history = record_history
       
       



    def close_vehicles_to(self, vehicle: 'kinematics.Vehicle', distance: float, count: int = None,
                          see_behind: bool = True) -> object:
        # print("vehicle.road, vehicle.position, heading, speed,target_lane_index, target_speed,route")
        #for i in self.vehicles:
        incrementStepSize()
        global floder
        global reputationincrements
        global finalPlausibilityAvg
        global checknewIteration
        global env
        global agent
        global action
        global observation
        global score
        global scores
        global plausibilityIncrements
        global centralizedRSURepIteration
        global centralizedRSUreputation
        global finalMassFunctionIterations
        global repDiffByRSU
        global ReputationByRsuToVehReport
        global rsuActionDiff
        global centralizedRSURepIteration
        global massfuncionsFromCentralizedReput
        global stepsize
        global centralizedcalculation
        global rewardFromFinal
        global falseMsgFromMaliciousVehicle
        global actionIterations
        global episode_count
        global smoothingFactor_space
        global n_steps
        global learn_iters
        global N
        global cc
        i=self.vehicles[0]
        coordinates=[]
        vehiclename={}
        diffvalues=[]
       
        faultVechicleposition={}
        newvehicleposition={}
        vehiclex=[]
        vehicley=[]
        f1=0
        global attackedVehicle
        vehicle_list = [i for i in range(len(i.road.vehicles))]
        #code to check new episode
        for ki in i.road.vehicles:
            if(len(self.vehicles)!=len(checknewIteration) and len(self.vehicles)>len(checknewIteration)):
                str1=str(ki)
                arr=str1.split(":")
                checknewIteration.append(arr[0])
            else:
                str1=str(ki)
                arr=str1.split(":")
                if(checknewIteration[f1]!=arr[0]):
                    attackedVehicle=[]

                    noisefa=int((len(i.road.vehicles)+1)/2)
                    # noisefa=noisefa%(len(i.road.vehicles)-1)
                    print("////////////////////////////////////////////////////////////////////////////////////////////\n\n//////////////////////",noisefa)
                   
                    for ite in range(noisefa):
                        #print("vehiclelist for picking attacker",vehicle_list)
                        a = choice(vehicle_list)
                        #print("vehicle picked for attacker",a)
                        attackedVehicle.append(a)
                        vehicle_list.remove(a)
                        # noisefa=random.randint(0,len(i.road.vehicles))
                        # noisefa=noisefa%(len(i.road.vehicles)-1)
                        # if(noisefa in attackedVehicle and noisefa!=(len(i.road.vehicles)-1)):
                        #     attackedVehicle.append(noisefa+1)
                    print("attacked vehicle",attackedVehicle)
                   
                   
                    #print("new iteration")
                    # plotReputation()
                    # plotPlausibility(plausibilityIncrements,"plausibilty","plausibility based on reports given by vehicle")
                    if(episode_count>100 and episode_count<150 ) or ( episode_count>200 and episode_count<250 ) or ( episode_count>500 and episode_count<750 ):

                        savePlotVals(centralizedRSURepIteration,"Centralized reputation","Centralized reputation based on validatation by RSU","centralized",floder+"centralized" )
                        savePlotVals(finalMassFunctionIterations,"Final reputation","The final reputation after 3 level dempster shafer implementation","final",floder+"final")
                        savePlotVals(actionIterations,"SmootingValues","The action graph","action",floder+"action")
                        savePlotVals(rewardValues,"Reward", "Reward Values","reward",floder+"reward")
                        # plotPlausibility(centralizedRSURepIteration,"Centralized reputation","Centralized reputation based on validatation by RSU","centralized" )
                        # plotPlausibility(finalMassFunctionIterations,"Final reputation","The final reputation after 3 level dempster shafer implementation","final")
                        # plotList(actionIterations,"SmootingValues","The action graph","action")
                    print("the centralized calls", centralizedcalculation)
                    resetReputation()
                    # time.sleep(10)

                    checknewIteration=[]
                    break
            f1+=1
       
        #to get only the  vehicle coordinates everytime in a dictionary
       
        for ki in i.road.vehicles:
            #print(s,type(s))
            arr1=[]
            arr1.append((ki.position[0],ki.position[1]))
            newvehicleposition[ki.id]=arr1
            #(x,y)=newvehicleposition[ki.id]
            #print("x,y",x,y)
            vehiclex.append(ki.position[0])
            vehicley.append(ki.position[1])
        # print("vehicle list",vehicle_list)
        # print("new vehicle list ",newvehicleposition)
        newfaultposition=dict(newvehicleposition)

       
       
       
       

        faultvehiclex=[]
        faultvehicley=[]
        #code for fault generation
        # for j in i.road.vehicles:
        #     noisefa=random.randint(0,98)
        #     noisefa=noisefa%(len(i.road.vehicles)-1)
        #     noise1=random.randint(0,9)
        #     if(j.id==noisefa):
        #         faultVechicleposition[j.id]=(j.position[0]+noise1,j.position[1]+noise1)
        #         # print("ok")
        #         (x,y)=faultVechicleposition[j.id]
        #         # print("x,y",x,y)
        #         faultvehiclex.append(x)
        #         faultvehicley.append(y)
        #         # print("type of x coridb",type(x))
        #         # print("type of faultvechile",type(faultVechicleposition[j.id]))
   
        #     else:
        #         faultVechicleposition[j.id]=(j.position[0],j.position[1])
        #         (x,y)=faultVechicleposition[j.id]
        #         faultvehiclex.append(x)
        #         faultvehicley.append(y)
        #         # print("type of x coridb",type(x))
        #         # print("type of faultvechile",type(faultVechicleposition[j.id]))
        #     coordinates.append((j.position[0],j.position[1]))
        #     vehiclename[j.id]=(j.position[0],j.position[1])
        #     print("ivehicle",i.position,"each vehicle position",j.position)
        # print("faultyy vehicle position",faultVechicleposition)
        #plotVehiclePosition(vehiclex,vehicley,faultvehiclex,faultvehicley)
        for j in i.road.vehicles:
            coordinates.append((j.position[0],j.position[1]))
            vehiclename[j.id]=(j.position[0],j.position[1])
        # print("vehiclename",vehiclename)
       
        # if(len(coordinates)<5):
        #     observedRange=60
        # else:
        #     observedRange=60
        adjacency_lists = defaultdict(set)
        adjacent={}
        for  ind, coord in enumerate(coordinates):
            count=0
            for other_coord in coordinates[ind:]:
                if(other_coord!=coord):
                    diffvalues.append(abs((coord[0] - other_coord[0])))
                    diffvalues.append(abs((coord[1] - other_coord[1])))
                    closekm=haversine(coord[0],coord[1],other_coord[0],other_coord[1])
                    # pythkm=pythagorus(coord[0],coord[1],other_coord[0],other_coord[1])
                    # print("coordinate and other ",ind,ind+count,coord,other_coord)
                    # print("the distance between vehicles",closekm)
                    # print("the distance b/w vehicles in pythagorus",pythkm)
                    # print("diff :","x",abs((coord[0] - other_coord[0])),"y",abs((coord[1] - other_coord[1])))
                    # if abs((coord[0] - other_coord[0])) <= observedRange and abs((coord[1] - other_coord[1])) <= observedRange:
                    if closekm<600:
                        adjacency_lists[coord].add(other_coord)
                        adjacency_lists[other_coord].add(coord)
                    count=count+1
                   
        matrix = [[0] * len(coordinates) for i in range(len(coordinates))]
        for i , j in adjacency_lists.items():
            arr1=[]
            for k in j:
                arr1.append(coordinates.index(k))
                matrix[coordinates.index(i)][coordinates.index(k)]=1
                #print("coordinates",coordinates.index(i),coordinates.index(k))
            adjacent[coordinates.index(i)]=arr1
               
        # print("adjacent veh",adjacent)
        #print("matrix",matrix)
        # print("newvehicleposition",newvehicleposition)
        # print("vehiclename",vehiclename)
        # "ATTACKER CODE ADDING NOISE TO EXISTING POSITIONS"
        peerrepobsm={}
        selfrepobsm={}
        # action, prob, val = agent.choose_action(observation)
          
        print("action",smoothingFactor_space[action])
        action=agent.choose_action(observation)
        for ke in vehiclename:
            a=0
            arra1=[]
            peerrepo="peerrepobsm"+str(ke)
            selfrepo="selfrepobsmcollect"+str(ke)
            selfrepobsm[selfrepo]={}
            peerrepobsm[peerrepo]={}
            selfrepoco=selfrepobsm[selfrepo]
            currentpeerdict=peerrepobsm[peerrepo]
            arra1.append(vehiclename[ke])
       
            
            
            for ko in vehiclename:
                if(matrix[ke][ko]==1): 
                    if((ke in attackedVehicle) and (smoothingFactor_space[action]<0.75) and (ko not in attackedVehicle)):
                           
                            # print("in attacked loop before adding noise",smoothingFactor_space[action])
                            # time.sleep(0.5)
                            arr=newvehicleposition[ke]
                            (x,y)=vehiclename[ko]
                            stepincre=repuationStepIncrement.get(ke,[])
                            selfrepoco[ko]=vehiclename[ko]
                            currentpeerdict[ko]=addWhiteGuassianNoise(vehiclename[ko])
                            # print("in attacked loop after adding noise")
                            # print("step increment\n",stepincre)
                            # noise1=0
                            # if(len(stepincre)>0 and abs(stepsize-stepincre[0])>10):
                                # noise1=random.randint(1,9)
                            #print("x,y",x,y)
                            # arr.append((x+noise1,y+noise1))
                            # arra1.append(faultVechicleposition[ko])
                            # newfaultposition[ke]=arra1
                            newvehicleposition[ke]=arr
                    else:
                        currentpeerdict[ko]=vehiclename[ko]
                        selfrepoco[ko]=vehiclename[ko]
                        # addWhiteGuassianNoise(vehiclename[ko])
                        arr=newvehicleposition[ke]
                        arr.append(vehiclename[ko])
                        newvehicleposition[ke]=arr
                else:
                    if(ke!=ko):
                        # currentpeerdict[ko]=vehiclename[ko]
                        # selfrepoco[ko]=vehiclename[ko]
                        a=+1
                        arr=newvehicleposition[ke]
                        arr.append(-1)
                        newvehicleposition[ke]=arr
        # print("peer report bsm \n",peerrepobsm)
        # print("slef report bsm\n", selfrepobsm)
        #print("attacked vehicle",attackedVehicle)
        # print("new vehicle position",newvehicleposition)
        # print("new vehicle faulty position",newfaultposition)
       
       
        print("attacked vehicle list",attackedVehicle)
        reputation=calculateDecentralizedReputation(peerrepobsm,selfrepobsm,smoothingFactor_space[action],matrix)
        # print("reputation",reputation)
        if(rsuActionDiff<6):
            # print(" the value of step is ",stepsize)
            # print(" \n inside the centralized calulation the difference is ", rsuActionDiff,"\n")
            centralizedRSUreputation=calculateCentralizedReputation(vehiclename,newvehicleposition,centralizedRSUreputation)
            repDiffByRSU=calculateVehilceReputationReport(centralizedRSUreputation, reputation,repDiffByRSU)
           
        # print("the rsuActionDiff value", rsuActionDiff)
 
        if(rsuActionDiff==5):
            centralizedcalculation[stepsize]=ReputationByRsuToVehReport
            calculateVehRepu(repDiffByRSU,ReputationByRsuToVehReport)
            cc=1
   
           
            # print("the reputation of centralized is ",ReputationByRsuToVehReport)
            # calculateMassFunctionFromCentralizedRpeort(ReputationByRsuToVehReport)
        # print("vehicles reputations",reputation,"\n")
        calculateMassFunctionFromCentralizedRpeort(ReputationByRsuToVehReport)
        avgReputation=calculateavgReputation(adjacent,reputation,matrix)
        # currReward=calculateReward(adjacent,reputation)
        # # print("the reward and reputation of vehicles",avgReputation,currReward)
        # observation_, reward, done , info =env.step(action,currReward,avgReputation)
        # score+=reward
        # agent.store_transition(observation, action, reward, observation_, done )
        # agent.learn()
        #         # observation= observation_
        # print("\n the mass function from centralizd rept is ",massfuncionsFromCentralizedReput)
        # print("the initial round mass functions are \n", massFunctionsFromLevel1Plausibility)
        # print("the second round mass functions are \n", massFunctionsFromLevel2Plausibility)
        combineMassFunctionsFromThreeLevels(massFunctionsFromLevel1Plausibility,massFunctionsFromLevel2Plausibility,massfuncionsFromCentralizedReput)
        if(cc==1):
            getTheMaliciousMassFunctionFromMassFunction(massFunctionsFromLevel1Plausibility,massFunctionsFromLevel2Plausibility,massfuncionsFromCentralizedReput,attackedVehicle,floder)
        # claculateGyawaliReward(mass)
        print("final malcios list",finalMaliciousVehicles)
        falseMsgFromMaliciousVehicle=0
        for i in finalMaliciousVehicles:
            repstr="reputationby"
            # print("the reputation form the vehicles are",repstr+str(i))
            repByMaliciousVehicle=reputation[repstr+str(i)]
            falseMsgFromMaliciousVehicle+=len(repByMaliciousVehicle)
        totalMsgFromVehicles=0
        for i in reputation:
            repstr="reputationby"
            repByVehicle=reputation[i]
            totalMsgFromVehicles+=len(repByVehicle)
        print("the false msg count and the total msgs are", falseMsgFromMaliciousVehicle,totalMsgFromVehicles)
        rewardFromFinal=(totalMsgFromVehicles-falseMsgFromMaliciousVehicle)/totalMsgFromVehicles
        #rewardFromFinal=-(falseMsgFromMaliciousVehicle/totalMsgFromVehicles)
        currReward=rewardFromFinal     #calculateReward(adjacent,reputation)
        rewardValues.append(currReward)
        # print("the reward and reputation of vehicles",avgReputation,currReward)
        
        # observation_, reward, done, info = env.step(action,currReward,round(finalPlausibilityAvg,2))
        # n_steps += 1
        # actor critic
        # observation_, reward, done , info =env.step(action,currReward,round(finalPlausibilityAvg,2))
        # score+=reward
        # agent.store_transition(observation, action, reward, observation_, done )
            
        # agent.store_transition(observation, action, reward, observation_, done ) #ppo/qlearning
        # agent.remember(observation, action, prob, val, reward, done) #ppo ppo
        # if n_steps % N == 0: ppo
        #         agent.learn()
        #         learn_iters += 1
        # agent.learn(observation, reward, observation_, done)
        # agent.learn(observation, reward, observation_, done)
# actor critic end
# dqn 
        observation_, reward, done , info =env.step(action,currReward,round(finalPlausibilityAvg,2))
        score+=reward
        agent.store_transition(observation, action, reward, observation_, done )
        agent.learn()
        observation= observation_
        actionIterations.append(smoothingFactor_space[action])










       
        #plot vehicle coordinates
       

       # print("idict",vehiclename)
        # print("faulty index",faultVechicleposition)
        #malicious=calculateReputation(newvehicleposition,matrix)
    #plotMalNorVehiclePosition(newvehicleposition,malicious,adjacent)
        # if(len(adjacent)>0):
        #     validation=validationPoints(vehiclename,malicious,adjacent)
        #     print(validation)
        #     validatePoints(validation,newvehicleposition,malicious,adjacent)
        #     plotSingleValidationpoints(validation,newvehicleposition,malicious,adjacent)

       

        # # print("peer faulty",peerfaultyreports)
        # print(",matrix",matrix)
       # print("cooridnates",coordinates)
        # print("size of the vehicles",len(coordinates))
        # print("adj",adjacency_lists,"adj end")
        # print(diffvalues)
        # print(self.vehicles[0].road)
       


        #     print("road",i.road,"position",i.position,"heading",i.heading,"speed",i.speed,"target lane",i.target_lane_index,"target speed",i.target_speed,"route",i.route)
       

        vehicles = [v for v in self.vehicles
                    if np.linalg.norm(v.position - vehicle.position) < distance
                    and v is not vehicle
                    and (see_behind or -2 * vehicle.LENGTH < vehicle.lane_distance_to(v))]

        vehicles = sorted(vehicles, key=lambda v: abs(vehicle.lane_distance_to(v)))
        if count:
            vehicles = vehicles[:count]
        return vehicles

    def act(self) -> None:
        """Decide the actions of each entity on the road."""
        # e.g., len(self.vehicles) = 7
        # if vehicle: IDMVehicle, it will go to the behavior.py
        # if vehicle: MDPVehicle, it will go to the behavior.py
        for vehicle in self.vehicles:  # all the vehicles on the road
            vehicle.act()

   

    def step(self, dt: float) -> None:
        """
        Step the dynamics of each entity on the road.

        :param dt: timestep [s]
        """
        for vehicle in self.vehicles:
            vehicle.step(dt)
        for vehicle in self.vehicles:
            for other in self.vehicles:
                vehicle.check_collision(other)
            for other in self.objects:
                vehicle.check_collision(other)

    def surrounding_vehicles(self, vehicle: 'kinematics.Vehicle', lane_index: LaneIndex = None) \
            -> Tuple[Optional['kinematics.Vehicle'], Optional['kinematics.Vehicle']]:
        """
        Find the preceding and following vehicles of a given vehicle.

        :param vehicle: the vehicle whose neighbours must be found
        :param lane_index: the lane on which to look for preceding and following vehicles.
                     It doesn't have to be the current vehicle lane but can also be another lane, in which case the
                     vehicle is projected on it considering its local coordinates in the lane.
        :return: its preceding vehicle, its following vehicle
        """
        lane_index = lane_index or vehicle.lane_index
        if not lane_index:
            return None, None
        s = vehicle.position[0]  # x position
        s_front = s_rear = None
        v_front = v_rear = None

        # we do not consider obstacles
        for v in self.vehicles:
            if v is not vehicle and not isinstance(v, Landmark):
                if lane_index == ("a", "b", 0) or lane_index == ("b", "c", 0) or lane_index == (
                        "c", "d", 0):
                    if lane_index == ("a", "b", 0) and (
                            v.lane_index == ("a", "b", 0) or v.lane_index == ("b", "c", 0)):
                        s_v, lat_v = v.position
                    elif lane_index == ("b", "c", 0) and (
                            v.lane_index == ("a", "b", 0) or v.lane_index == ("b", "c", 0) or v.lane_index == (
                    "c", "d", 0)):
                        s_v, lat_v = v.position
                    elif lane_index == ("c", "d", 0) and (v.lane_index == ("b", "c", 0) or v.lane_index == (
                    "c", "d", 0)):
                        s_v, lat_v = v.position
                    else:
                        continue
                else:
                    if lane_index == ("j", "k", 0) and (
                            v.lane_index == ("j", "k", 0) or v.lane_index == ("k", "b", 0)):
                        s_v, lat_v = v.position
                    elif lane_index == ("k", "b", 0) and (
                            v.lane_index == ("j", "k", 0) or v.lane_index == ("k", "b", 0) or v.lane_index == (
                    "b", "c", 1)):
                        s_v, lat_v = v.position
                    elif lane_index == ("b", "c", 1) and (
                            v.lane_index == ("k", "b", 0) or v.lane_index == (
                    "b", "c", 1)):
                        s_v, lat_v = v.position
                    else:
                        continue

                if s <= s_v and (s_front is None or s_v <= s_front):
                    s_front = s_v
                    v_front = v
                if s_v < s and (s_rear is None or s_v > s_rear):
                    s_rear = s_v
                    v_rear = v
        return v_front, v_rear

    def neighbour_vehicles(self, vehicle: 'kinematics.Vehicle', lane_index: LaneIndex = None) \
            -> Tuple[Optional['kinematics.Vehicle'], Optional['kinematics.Vehicle']]:
        """
        Find the preceding and following vehicles of a given vehicle.
        :param vehicle: the vehicle whose neighbours must be found
        :param lane_index: the lane on which to look for preceding and following vehicles.
                     It doesn't have to be the current vehicle lane but can also be another lane, in which case the
                     vehicle is projected on it considering its local coordinates in the lane.
        :return: its preceding vehicle, its following vehicle
        """
        lane_index = lane_index or vehicle.lane_index
        if not lane_index:
            return None, None
        lane = self.network.get_lane(lane_index)
        s = self.network.get_lane(lane_index).local_coordinates(vehicle.position)[0]
        s_front = s_rear = None
        v_front = v_rear = None
        for v in self.vehicles + self.objects:
            if v is not vehicle and not isinstance(v, Landmark):  # self.network.is_connected_road(v.lane_index,
                # lane_index, same_lane=True):
                s_v, lat_v = lane.local_coordinates(v.position)
                if not lane.on_lane(v.position, s_v, lat_v, margin=1):
                    continue
                if s <= s_v and (s_front is None or s_v <= s_front):
                    s_front = s_v
                    v_front = v
                if s_v < s and (s_rear is None or s_v > s_rear):
                    s_rear = s_v
                    v_rear = v
        return v_front, v_rear

    def __repr__(self):
        return self.vehicles.__repr__()