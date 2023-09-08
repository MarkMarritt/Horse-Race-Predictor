import numpy as np
import scipy
import ast
import pandas as pd
import warnings
warnings.filterwarnings("ignore")

def confidenceScaler(runs, wins):
    """Returns a scaled score of the rate based on how many runs have occured and the rate
    Runs, rate should be list/df
    outputs are both np arrays"""
    runs, wins = np.array(runs), np.array(wins)
    rate = wins / runs
    rate = np.nan_to_num(rate)
    avg = rate.mean()
    scaled = 0.5*(np.power((rate/avg), 0.5*np.log(runs)))
    scaled = np.nan_to_num(scaled, posinf = 0, neginf = 0)
    return rate, scaled

def rateTrends(rate14, rate30, rate60):
    """returns 1st and second differences of the horse rates, should all be arrays"""
    diff1 = rate14 - rate30
    diff2 = diff1 - (rate30 - rate60)
    return diff1, diff2


#may need to be changed if different events suit different ages
def ageScaler(age):
    """returns a scaled age """
    x = age
    #assuming optimal age is 4.5 years:
    return 0.2*(np.sin(0.5*(6.5-x))) + 0.857 - np.power(0.2*(6.5-x), 4)

def runsSinceScaler(runsSince):
    """returns a scaled time since gelded, should be in terms of runs. (Months could also work) """
    x = runsSince
    return (0.9/(np.sqrt(np.log10(0.5*x +2)))) - 0.65


def placesRegression(df):
    """does a linear regression of a column in a dataframe, returns next values of each"""
    regColumn = []
    for row in df:
        if isinstance(row, str):
            row = ast.literal_eval(row)
        if not isinstance(row, list):
            regColumn.append(0)
            continue
        x = len(row)
        if x == 0:
            regColumn.append(0)
        else:
            X = np.arange(x)
            slope, intercept, r_value, p_value, std_err = scipy.stats.linregress(X, row)
            pred = slope*(x) + intercept
            if np.isnan(pred):
                regColumn.append(0)
            else:
                regColumn.append(pred)
    return np.array(regColumn)


def clean_df(df, binaryPositions = False):
    """cleans data, removing any na values and then creates new columns which are derived from the other columns"""
    df['Penalty'].fillna(0, inplace=True)
    cols = ["SameJockey","ORWinner","ORWins","ORPlaced","HighClassWin", "GradeWinner", 
            "TopRTypeJockey", "LastBestSpeed", "LastBestSpeed3", "LastBestForm",  "LastBestForm3",
            "FutureEntry", "LastTimeWin", "NewTrain", "Noted", "SameCourse", "SameCourse_2",
            "HighGradeWinner", "TopRTypeTrainer", "Select", "Short"]
    zeros = ["HWM", "Travelled", "LastWM", "MinOdds", "MaxOdds", "AvgOdds", "LastWon", 
            "HcapCount", "OR", "Last_OR", "Form", "Speed", "OddsPos", "Wins", "Runs",
            "HCrsWin", "HCrsRun","HGngWin", "HGngRun", "HDisWin", "HDisRun", "HClassWin", "HClassRun",
            "TCrseWin", "TCrseRun", "TRTypeWin", "TRTypeRun", "T14Win", "T14Run", "T30Win",
            "T30Run", "T60Win", "T60Run", "TJWin", "TJRun", "JCrseWin", "JCrseRun", "JRTypeWin",
            "JRTypeRun", "J14Win", "J14Run","J30Win", "J30Run", "J60Win", "J60Run", "Fitness", "Ability", 
            "Conditions", "Market", "Vibes","OCrseWin", "OCrseRun", "ORTypeWin",
            "ORTypeRun", "O14Win", "O14Run", "O30Win", "O30Run", "O60Win", "O60Run", "SCrseWin",
            "SCrseRun", "SRTypeWin", "SRTypeRun", "S14Win", "S14Run", "S30Win", "S30Run", "S60Win", "S60Run",
            "HHcapWin", "HHcapRun", "JBreakWins", "JBreakRuns", "JDebutWins", "JDebutRuns", "JHGBackWins",
            "JHGBackRuns", "JHGFirstWins", "JHGFirstRuns", "JHGHcp1sttWins", "JHGHcp1sttRuns", "JFavWin",
            "JFavRuns", "JOOWin", "JOORuns", "JE2Runs", "JE2Win", "J23Runs", "J23Win", "J35Win", "J35Runs", 
            "J59Win", "J59Runs", "J916Win", "J916Runs", "J16PlusWin", "J16PlusRuns", "TBreakWins", "TBreakRuns", 
            "TDebutWins", "TDebutRuns", "THGBackWins", "THGBackRuns", "THGFirstWins", "THGFirstRuns", 
            "THGHcp1sttWins", "THGHcp1sttRuns", "TFavWin", "TFavRuns", "TOOWin", "TOORuns", "TE2Win", "TE2Runs",
            "T23Win", "T23Runs", "T35Win", "T35Runs", "T59Win", "T59Runs", "T916Win", "T916Runs", "T16PlusWin",
            "T16PlusRuns", "SRGoingWin", "SRGoingRun", "SRDistanceWin", "SRDistanceRun", "SRClassWin", "SRClassRun",
            "Place1", "Place2", "Place3", "Place4", "Place5", "Place6",
            "LastPerf", "LastPerf_2", "FormImproved", "SpeedImproved", "ORImproved", "ShrewdTrainer",
            "BrkRuns", "Tips", "SubType" ]
    # df.drop(df[df['Performance'] == 0].index, inplace=True)
    df[cols] = df[cols].replace({True: 1, False: 0})
    df[zeros] = df[zeros].replace({np.nan: 0})
    df['ShrewdTrainer'].replace({'Y': 1, 'N': 0}, inplace=True)
    df['Gambled'].replace({'G': 1, np.nan: 0}, inplace=True)
    df['CatGradeWinner'].replace({'BOTH': 3, 'HIGH': 2, 'SAME': 1, np.nan: 0}, inplace=True)
    df['WellHcap'].replace({'negative': 1, 'neutral': 2, 'positive': 3, 'empty': 0, np.nan: 0}, inplace=True)
    df['CatClassWinner'].replace({'BOTH': 3, 'HIGH': 2, 'SAME': 1, np.nan: 0}, inplace=True)
    df['TForm'].replace({'C': 0, 'N': 1, 'H': 2}, inplace=True)
    df['JForm'].replace({'C': 0, 'N': 1, 'H': 2}, inplace=True)
    #df["Position"] = np.where(df["Position"] != 1, 0, 1)
    # df[df.isna().any(axis=1)]
    #marks changes
    df["Position"].replace({np.nan: 30.0},inplace = True)
    df["Rank"].replace({np.nan: 20.0},inplace = True)
    df["RaceType"].replace({"MAIDEN" : 0, "HANDICAP" : 1, "GRADED" : 2, 'SELLING' : 3, 'LISTED' : 4, 'CLAIMING' : 5, np.nan: 6, "OTHER" : 6  }, inplace = True)
    df["FormRank"].replace({np.nan:30.0}, inplace = True)
    df["SpeedRank"].replace({np.nan:30.0}, inplace = True)
    df["ORRank"].replace({np.nan:30.0}, inplace = True)
    df["WeightRank"].replace({np.nan:30.0}, inplace = True)
    df["Last6JCPred"] = placesRegression(df["Last6JC"])
    df["Last6FormPred"] = placesRegression(df["Last6Form"])
    df["Last6SpeedPred"] = placesRegression(df["Last6Speed"])
    df["Place2"] = df["Place2"].div(2)
    df["Place3"] = df["Place3"].div(3)
    df["Place4"] = df["Place4"].div(4)
    df["Place5"] = df["Place5"].div(5)
    df["Place6"] = df["Place6"].div(6)
    df["SpeedRankFrac"] = df["SpeedRank"].div(df["DecRunners"])
    df["FormRankFrac"] = df["FormRank"].div(df["DecRunners"])
    df["RankFrac"] = df["Rank"].div(df["DecRunners"])
    df["RelTips"] = df["Tips"].mul(df["DecRunners"])
    df["Last6Place"] = df[["Place6", "Place5", "Place4", "Place3", "Place2", "Place1"]].values.tolist()
    df["Last6PlacePred"] = placesRegression(df["Last6Place"])
    df = df.drop(["Last6Place", "Last6Speed", "Last6Form", "Last6JC"], axis = "columns") # these are list types and cant do anything with
    df["ScaledAge"] = ageScaler(df["Age"])
    df["LastPerfDiff"] = df["LastPerf"] - df["LastPerf_2"]
    df["WinRate"], df["WinRateScaled"] = confidenceScaler(df["HCrsRun"], df["HCrsWin"])
    df["HCrsWinRate"], df["HCrsWinRateScaled"] = confidenceScaler(df["HCrsRun"], df["HCrsWin"])
    df["HGngWinRate"], df["HGngWinRateScaled"] = confidenceScaler(df["HGngRun"], df["HGngWin"])
    df["HDisWinRate"], df["HDisWinRateScaled"] = confidenceScaler(df["HDisRun"], df["HDisWin"])
    df["HClassWinRate"], df["HClassWinRateScaled"] = confidenceScaler(df["HClassRun"], df["HClassWin"])
    df["TCrseWinRate"], df["TCrseWinRateScaled"] = confidenceScaler(df["TCrseRun"], df["TCrseWin"])
    df["TRTypeWinRate"], df["TRTypeWinRateScaled"] = confidenceScaler(df["TRTypeRun"], df["TRTypeWin"])
    df["T14WinRate"], df["T14WinRateScaled"] = confidenceScaler(df["T14Run"], df["T14Win"])
    df["T30WinRate"], df["T30WinRateScaled"] = confidenceScaler(df["T30Run"], df["T30Win"])
    df["T60WinRate"], df["T60WinRateScaled"] = confidenceScaler(df["T60Run"], df["T60Win"])
    df["TJWinRate"], df["TJWinRateScaled"] = confidenceScaler(df["TJRun"], df["TJWin"])
    df["JCrseWinRate"], df["JCrseWinRateScaled"] = confidenceScaler(df["JCrseRun"], df["JCrseWin"])
    df["JRTypeWinRate"], df["JRTypeWinRateScaled"] = confidenceScaler(df["JRTypeRun"], df["JRTypeWin"])
    df["J14WinRate"], df["J14WinRateScaled"] = confidenceScaler(df["J14Run"], df["J14Win"])
    df["J30WinRate"], df["J30WinRateScaled"] = confidenceScaler(df["J30Run"], df["J30Win"])
    df["J60WinRate"], df["J60WinRateScaled"] = confidenceScaler(df["J60Run"], df["J60Win"])
    df["OCrseWinRate"], df["OCrseWinRateScaled"] = confidenceScaler(df["OCrseRun"], df["OCrseWin"])
    df["ORTypeWinRate"], df["ORTypeWinRateScaled"] = confidenceScaler(df["ORTypeRun"], df["ORTypeWin"])
    df["O14WinRate"], df["O14WinRateScaled"] = confidenceScaler(df["O14Run"], df["O14Win"])
    df["O30WinRate"], df["O30WinRateScaled"] = confidenceScaler(df["O30Run"], df["O30Win"])
    df["O60WinRate"], df["O60WinRateScaled"] = confidenceScaler(df["O60Run"], df["O60Win"])
    df["SCrseWinRate"], df["SCrseWinRateScaled"] = confidenceScaler(df["SCrseRun"], df["SCrseWin"])
    df["SRTypeWinRate"], df["SRTypeWinRateScaled"] = confidenceScaler(df["SRTypeRun"], df["SRTypeWin"])
    df["S14WinRate"], df["S14WinRateScaled"] = confidenceScaler(df["S14Run"], df["S14Win"])
    df["S30WinRate"], df["S30WinRateScaled"] = confidenceScaler(df["S30Run"], df["S30Win"])
    df["S60WinRate"], df["S60WinRateScaled"] = confidenceScaler(df["S60Run"], df["S60Win"])
    df["HHcapWinRate"], df["HHcapWinRateScaled"] = confidenceScaler(df["HHcapRun"], df["HHcapWin"])
    df["JBreakWinRate"], df["JBreakWinRateScaled"] = confidenceScaler(df["JBreakRuns"], df["JBreakWins"])
    df["JDebutWinRate"], df["JDebutWinRateScaled"] = confidenceScaler(df["JDebutRuns"], df["JDebutWins"])
    df["JHGBackWinRate"], df["JHGBackWinRateScaled"] = confidenceScaler(df["JHGBackRuns"], df["JHGBackWins"])
    df["JHGFirstWinRate"], df["JHGFirstWinRateScaled"] = confidenceScaler(df["JHGFirstRuns"], df["JHGFirstWins"])
    df["JHGHcp1sttWinRate"], df["JHGHcp1sttWinRateScaled"] = confidenceScaler(df["JHGHcp1sttRuns"], df["JHGHcp1sttWins"])
    df["JFavWinRate"], df["JFavWinRateScaled"] = confidenceScaler(df["JFavRuns"], df["JFavWin"])
    df["JOOWinRate"], df["JOOWinRateScaled"] = confidenceScaler(df["JOORuns"], df["JOOWin"])
    df["JE2WinRate"], df["JE2WinRateScaled"] = confidenceScaler(df["JE2Runs"], df["JE2Win"])
    df["J23WinRate"], df["J23WinRateScaled"] = confidenceScaler(df["J23Runs"], df["J23Win"])
    df["J35WinRate"], df["J35WinRateScaled"] = confidenceScaler(df["J35Runs"], df["J35Win"])
    df["J59WinRate"], df["J59WinRateScaled"] = confidenceScaler(df["J59Runs"], df["J59Win"])
    df["J916WinRate"], df["J916WinRateScaled"] = confidenceScaler(df["J916Runs"], df["J916Win"])
    df["J16PlusWinRate"], df["J16PlusWinRateScaled"] = confidenceScaler(df["J16PlusRuns"], df["J16PlusWin"])
    df["TBreakWinRate"], df["TBreakWinRateScaled"] = confidenceScaler(df["TBreakRuns"], df["TBreakWins"])
    df["TDebutWinRate"], df["TDebutWinRateScaled"] = confidenceScaler(df["TDebutRuns"], df["TDebutWins"])
    df["THGBackWinRate"], df["THGBackWinRateScaled"] = confidenceScaler(df["THGBackRuns"], df["THGBackWins"])
    df["THGFirstWinRate"], df["THGFirstWinRateScaled"] = confidenceScaler(df["THGFirstRuns"], df["THGFirstWins"])
    df["THGHcp1sttWinRate"], df["THGHcp1sttWinRateScaled"] = confidenceScaler(df["THGHcp1sttRuns"], df["THGHcp1sttWins"])
    df["TFavWinRate"], df["TFavWinRateScaled"] = confidenceScaler(df["TFavRuns"], df["TFavWin"])
    df["TOOWinRate"], df["TOOWinRateScaled"] = confidenceScaler(df["TOORuns"], df["TOOWin"])
    df["TE2WinRate"], df["TE2WinRateScaled"] = confidenceScaler(df["TE2Runs"], df["TE2Win"])
    df["T23WinRate"], df["T23WinRateScaled"] = confidenceScaler(df["T23Runs"], df["T23Win"])
    df["T35WinRate"], df["T35WinRateScaled"] = confidenceScaler(df["T35Runs"], df["T35Win"])
    df["T59WinRate"], df["T59WinRateScaled"] = confidenceScaler(df["T59Runs"], df["T59Win"])
    df["T916WinRate"], df["T916WinRateScaled"] = confidenceScaler(df["T916Runs"], df["T916Win"])
    df["T16PlusWinRate"], df["T16PlusWinRateScaled"] = confidenceScaler(df["T16PlusRuns"], df["T16PlusWin"])
    df["SRGoingWinRate"], df["SRGoingWinRateScaled"] = confidenceScaler(df["SRGoingRun"], df["SRGoingWin"])
    df["SRDistanceWinRate"], df["SRDistanceWinRateScaled"] = confidenceScaler(df["SRDistanceRun"], df["SRDistanceWin"])
    df["SRClassWinRate"], df["SRClassWinRateScaled"] = confidenceScaler(df["SRClassRun"], df["SRClassWin"])
    df["TRateDiff1"], df["TRateDiff2"] = rateTrends(df["T14WinRate"], df["T30WinRate"], df["T60WinRate"])
    df["JRateDiff1"], df["JRateDiff2"] = rateTrends(df["J14WinRate"], df["J30WinRate"], df["J60WinRate"])
    df["ORateDiff1"], df["ORateDiff2"] = rateTrends(df["O14WinRate"], df["O30WinRate"], df["O60WinRate"])
    df["SRateDiff1"], df["SRateDiff2"] = rateTrends(df["S14WinRate"], df["S30WinRate"], df["S60WinRate"])
    tempPosition = df["Position"]
    df = df.drop(["Position","RaceCode","SubType", "Unnamed: 0"], axis = "columns")
    df["Position"] = tempPosition
    df = df.copy()


    if binaryPositions == True:
        df.loc[df["Position"] > 1, "Position"] = 0

    return df
