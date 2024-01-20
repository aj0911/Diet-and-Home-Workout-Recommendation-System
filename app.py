import pandas as pd
import numpy as np
from sklearn.cluster import KMeans
import pickle as pkl
import streamlit as st

data=pd.read_csv('input.csv')
Breakfastdata=data['Breakfast']
BreakfastdataNumpy=Breakfastdata.to_numpy()

Lunchdata=data['Lunch']
LunchdataNumpy=Lunchdata.to_numpy()

Dinnerdata=data['Dinner']
DinnerdataNumpy=Dinnerdata.to_numpy()

Food_itemsdata=data['Food_items']
breakfastfoodseparated=[]
Lunchfoodseparated=[]
Dinnerfoodseparated=[]

breakfastfoodseparatedID=[]
LunchfoodseparatedID=[]
DinnerfoodseparatedID=[]

for i in range(len(Breakfastdata)):
    if BreakfastdataNumpy[i]==1:
        breakfastfoodseparated.append(Food_itemsdata[i])
        breakfastfoodseparatedID.append(i)
    if LunchdataNumpy[i]==1:
        Lunchfoodseparated.append(Food_itemsdata[i])
        LunchfoodseparatedID.append(i)
    if DinnerdataNumpy[i]==1:
        Dinnerfoodseparated.append(Food_itemsdata[i])
        DinnerfoodseparatedID.append(i)
    
def Weight_Loss(age,weight,height,res):  
    LunchfoodseparatedIDdata = data.iloc[LunchfoodseparatedID]
    LunchfoodseparatedIDdata = LunchfoodseparatedIDdata.T
    val=list(np.arange(5,16))
    Valapnd=[0]+[4]+val
    LunchfoodseparatedIDdata=LunchfoodseparatedIDdata.iloc[Valapnd]
    LunchfoodseparatedIDdata=LunchfoodseparatedIDdata.T
    
    breakfastfoodseparatedIDdata = data.iloc[breakfastfoodseparatedID]
    breakfastfoodseparatedIDdata=breakfastfoodseparatedIDdata.T
    val=list(np.arange(5,16))
    Valapnd=[0]+[4]+val
    breakfastfoodseparatedIDdata=breakfastfoodseparatedIDdata.iloc[Valapnd]
    breakfastfoodseparatedIDdata=breakfastfoodseparatedIDdata.T
    
    DinnerfoodseparatedIDdata = data.iloc[DinnerfoodseparatedID]
    DinnerfoodseparatedIDdata=DinnerfoodseparatedIDdata.T
    val=list(np.arange(5,16))
    Valapnd=[0]+[4]+val
    DinnerfoodseparatedIDdata=DinnerfoodseparatedIDdata.iloc[Valapnd]
    DinnerfoodseparatedIDdata=DinnerfoodseparatedIDdata.T
    
    bmi = weight/(height**2) 
    
    for lp in range (0,80,20):
        test_list=np.arange(lp,lp+20)
        for i in test_list: 
            if(i == age):
                agecl=round(lp/20)    
    if ( bmi < 16):
        clbmi=4
    elif ( bmi >= 16 and bmi < 18.5):
        clbmi=3
    elif ( bmi >= 18.5 and bmi < 25):
        clbmi=2
    elif ( bmi >= 25 and bmi < 30):
        clbmi=1
    elif ( bmi >=30):
        clbmi=0
    
    DinnerfoodseparatedIDdata=DinnerfoodseparatedIDdata.to_numpy()
    LunchfoodseparatedIDdata=LunchfoodseparatedIDdata.to_numpy()
    breakfastfoodseparatedIDdata=breakfastfoodseparatedIDdata.to_numpy()
    ti=(clbmi+agecl)/2
    
    ## K-Means Based  Dinner Food
    Datacalorie=DinnerfoodseparatedIDdata[1:,1:len(DinnerfoodseparatedIDdata)]
    X = np.array(Datacalorie)
    kmeans = KMeans(n_clusters=3, random_state=0).fit(X)
    XValu=np.arange(0,len(kmeans.labels_))
    dnrlbl=kmeans.labels_
    
    ## K-Means Based  Lunch Food
    Datacalorie=LunchfoodseparatedIDdata[1:,1:len(LunchfoodseparatedIDdata)]
    X = np.array(Datacalorie)
    kmeans = KMeans(n_clusters=3, random_state=0).fit(X)
    XValu=np.arange(0,len(kmeans.labels_))
    lnchlbl=kmeans.labels_
    
    ## K-Means Based  Breakfast Food
    Datacalorie=breakfastfoodseparatedIDdata[1:,1:len(breakfastfoodseparatedIDdata)]
    X = np.array(Datacalorie)
    kmeans = KMeans(n_clusters=3, random_state=0).fit(X)
    XValu=np.arange(0,len(kmeans.labels_))
    brklbl=kmeans.labels_
    
    ## Reading of the Dataset
    datafin=pd.read_csv('inputfin.csv')
    
    dataTog=datafin.T

    bmicls=[0,1,2,3,4]
    agecls=[0,1,2,3,4]
    
    weightlosscat = dataTog.iloc[[1,2,7,8]]
    weightlosscat=weightlosscat.T
    weightgaincat= dataTog.iloc[[0,1,2,3,4,7,9,10]]
    weightgaincat=weightgaincat.T
    healthycat = dataTog.iloc[[1,2,3,4,6,7,9]]
    healthycat=healthycat.T
    weightlosscatDdata=weightlosscat.to_numpy()
    weightgaincatDdata=weightgaincat.to_numpy()
    healthycatDdata=healthycat.to_numpy()
    weightlosscat=weightlosscatDdata[1:,0:len(weightlosscatDdata)]
    weightgaincat=weightgaincatDdata[1:,0:len(weightgaincatDdata)]
    healthycat=healthycatDdata[1:,0:len(healthycatDdata)]
    
    weightlossfin=np.zeros((len(weightlosscat)*5,6),dtype=np.float32)
    
    t=0
    r=0
    s=0
    yt=[]
    yr=[]
    ys=[]
    for zz in range(5):
        for jj in range(len(weightlosscat)):
            valloc=list(weightlosscat[jj])
            valloc.append(bmicls[zz])
            valloc.append(agecls[zz])
            weightlossfin[t]=np.array(valloc)
            yt.append(brklbl[jj])
            t+=1
            
        for jj in range(len(weightlosscat)):
            valloc=list(weightlosscat[jj])
            valloc.append(bmicls[zz])
            valloc.append(agecls[zz])
            weightlossfin[r]=np.array(valloc)
            yr.append(lnchlbl[jj])
            r+=1
            
        for jj in range(len(weightlosscat)):
            valloc=list(weightlosscat[jj])
            valloc.append(bmicls[zz])
            valloc.append(agecls[zz])
            weightlossfin[s]=np.array(valloc)
            ys.append(dnrlbl[jj])
            s+=1
            
    X_test=np.zeros((len(weightlosscat),6),dtype=np.float32)
    
    for jj in range(len(weightlosscat)):
        valloc=list(weightlosscat[jj])
        valloc.append(agecl)
        valloc.append(clbmi)
        X_test[jj]=np.array(valloc)*ti
        
    if res==1:
        X_train= weightlossfin
        y_train=yt
        
    elif res==2:
        X_train= weightlossfin
        y_train=yr 
        
    elif res==3:
        X_train= weightlossfin
        y_train=ys
        
    from sklearn.ensemble import RandomForestClassifier
    
    clf=RandomForestClassifier(n_estimators=100)
    
    clf.fit(X_train,y_train)
    
    y_pred=clf.predict(X_test)
    if(res==1):
        num = 2;
    elif(res==2):
        num = 1;
    elif(res==3):
        num=2;
    return_list_data = [];
    for ii in range(len(y_pred)):
        if y_pred[ii]==num:
            return_list_data.append(Food_itemsdata[ii])
    return return_list_data;
            
def Weight_Gain(age,weight,height,res):
    LunchfoodseparatedIDdata = data.iloc[LunchfoodseparatedID]
    LunchfoodseparatedIDdata=LunchfoodseparatedIDdata.T
    val=list(np.arange(5,15))
    Valapnd=[0]+val
    LunchfoodseparatedIDdata=LunchfoodseparatedIDdata.iloc[Valapnd]
    LunchfoodseparatedIDdata=LunchfoodseparatedIDdata.T
    
    breakfastfoodseparatedIDdata = data.iloc[breakfastfoodseparatedID]
    breakfastfoodseparatedIDdata=breakfastfoodseparatedIDdata.T
    val=list(np.arange(5,15))
    Valapnd=[0]+val
    breakfastfoodseparatedIDdata=breakfastfoodseparatedIDdata.iloc[Valapnd]
    breakfastfoodseparatedIDdata=breakfastfoodseparatedIDdata.T
    
    
    DinnerfoodseparatedIDdata = data.iloc[DinnerfoodseparatedID]
    DinnerfoodseparatedIDdata=DinnerfoodseparatedIDdata.T
    val=list(np.arange(5,15))
    Valapnd=[0]+val
    DinnerfoodseparatedIDdata=DinnerfoodseparatedIDdata.iloc[Valapnd]
    DinnerfoodseparatedIDdata=DinnerfoodseparatedIDdata.T

    bmi = weight/(height**2) 
    agewiseinp=0
    
    for lp in range (0,80,20):
        test_list=np.arange(lp,lp+20)
        for i in test_list: 
            if(i == age):
                tr=round(lp/20)  
                agecl=round(lp/20)    

    if ( bmi < 16):
        clbmi=4
    elif ( bmi >= 16 and bmi < 18.5):
        clbmi=3
    elif ( bmi >= 18.5 and bmi < 25):
        clbmi=2
    elif ( bmi >= 25 and bmi < 30):
        clbmi=1
    elif ( bmi >=30):
        clbmi=0    
    val1=DinnerfoodseparatedIDdata.describe()
    valTog=val1.T
    DinnerfoodseparatedIDdata=DinnerfoodseparatedIDdata.to_numpy()
    LunchfoodseparatedIDdata=LunchfoodseparatedIDdata.to_numpy()
    breakfastfoodseparatedIDdata=breakfastfoodseparatedIDdata.to_numpy()
    ti=(bmi+agecl)/2
    
    ## K-Means Based  Dinner Food
    Datacalorie=DinnerfoodseparatedIDdata[1:,1:len(DinnerfoodseparatedIDdata)]
    X = np.array(Datacalorie)
    kmeans = KMeans(n_clusters=3, random_state=0).fit(X)
    XValu=np.arange(0,len(kmeans.labels_))
    dnrlbl=kmeans.labels_
    
    ## K-Means Based  lunch Food
    Datacalorie=LunchfoodseparatedIDdata[1:,1:len(LunchfoodseparatedIDdata)]
    X = np.array(Datacalorie)
    kmeans = KMeans(n_clusters=3, random_state=0).fit(X)
    XValu=np.arange(0,len(kmeans.labels_))
    lnchlbl=kmeans.labels_
    
    ## K-Means Based  lunch Food
    Datacalorie=breakfastfoodseparatedIDdata[1:,1:len(breakfastfoodseparatedIDdata)]
    X = np.array(Datacalorie)
    kmeans = KMeans(n_clusters=3, random_state=0).fit(X)
    XValu=np.arange(0,len(kmeans.labels_))
    brklbl=kmeans.labels_
    
    datafin=pd.read_csv('inputfin.csv')
    datafin.head(5)
    dataTog=datafin.T
    bmicls=[0,1,2,3,4]
    agecls=[0,1,2,3,4]
    weightlosscat = dataTog.iloc[[1,2,7,8]]
    weightlosscat=weightlosscat.T
    weightgaincat= dataTog.iloc[[0,1,2,3,4,7,9,10]]
    weightgaincat=weightgaincat.T
    healthycat = dataTog.iloc[[1,2,3,4,6,7,9]]
    healthycat=healthycat.T
    weightlosscatDdata=weightlosscat.to_numpy()
    weightgaincatDdata=weightgaincat.to_numpy()
    healthycatDdata=healthycat.to_numpy()
    weightlosscat=weightlosscatDdata[1:,0:len(weightlosscatDdata)]
    weightgaincat=weightgaincatDdata[1:,0:len(weightgaincatDdata)]
    healthycat=healthycatDdata[1:,0:len(healthycatDdata)]
    weightgainfin=np.zeros((len(weightgaincat)*5,10),dtype=np.float32)
    t=0
    r=0
    s=0
    yt=[]
    yr=[]
    ys=[]
    for zz in range(5):
        for jj in range(len(weightgaincat)):
            valloc=list(weightgaincat[jj])
            valloc.append(bmicls[zz])
            valloc.append(agecls[zz])
            weightgainfin[t]=np.array(valloc)
            yt.append(brklbl[jj])
            t+=1
        for jj in range(len(weightgaincat)):
            valloc=list(weightgaincat[jj])
            valloc.append(bmicls[zz])
            valloc.append(agecls[zz])
            weightgainfin[r]=np.array(valloc)
            yr.append(lnchlbl[jj])
            r+=1
        for jj in range(len(weightgaincat)):
            valloc=list(weightgaincat[jj])
            valloc.append(bmicls[zz])
            valloc.append(agecls[zz])
            weightgainfin[s]=np.array(valloc)
            ys.append(dnrlbl[jj])
            s+=1

    
    X_test=np.zeros((len(weightgaincat),10),dtype=np.float32)

   
    for jj in range(len(weightgaincat)):
        valloc=list(weightgaincat[jj])
        valloc.append(agecl)
        valloc.append(clbmi)
        X_test[jj]=np.array(valloc)*ti
    
    if res==1:
        X_train= weightgainfin
        y_train=yt
        
    elif res==2:
        X_train= weightgainfin
        y_train=yr 
        
    elif res==3:
        X_train= weightgainfin
        y_train=ys
      
    from sklearn.ensemble import RandomForestClassifier
    
    clf=RandomForestClassifier(n_estimators=100)
    
    clf.fit(X_train,y_train)
    
    y_pred=clf.predict(X_test)
    if(res==1):
        num = 2;
    elif(res==2):
        num = 1;
    elif(res==3):
        num=2;
    return_list_data = [];
    for ii in range(len(y_pred)):
        if y_pred[ii]==num:
            return_list_data.append(Food_itemsdata[ii])
    return return_list_data;
            
def Healthy(age,weight,height,res):
    LunchfoodseparatedIDdata = data.iloc[LunchfoodseparatedID]
    LunchfoodseparatedIDdata=LunchfoodseparatedIDdata.T
    val=list(np.arange(5,15))
    Valapnd=[0]+val
    LunchfoodseparatedIDdata=LunchfoodseparatedIDdata.iloc[Valapnd]
    LunchfoodseparatedIDdata=LunchfoodseparatedIDdata.T
    
    breakfastfoodseparatedIDdata = data.iloc[breakfastfoodseparatedID]
    breakfastfoodseparatedIDdata=breakfastfoodseparatedIDdata.T
    val=list(np.arange(5,15))
    Valapnd=[0]+val
    breakfastfoodseparatedIDdata=breakfastfoodseparatedIDdata.iloc[Valapnd]
    breakfastfoodseparatedIDdata=breakfastfoodseparatedIDdata.T
    
    DinnerfoodseparatedIDdata = data.iloc[DinnerfoodseparatedID]
    DinnerfoodseparatedIDdata=DinnerfoodseparatedIDdata.T
    val=list(np.arange(5,15))
    Valapnd=[0]+val
    DinnerfoodseparatedIDdata=DinnerfoodseparatedIDdata.iloc[Valapnd]
    DinnerfoodseparatedIDdata=DinnerfoodseparatedIDdata.T
    bmi = weight/(height**2) 
    agewiseinp=0
    
    for lp in range (0,80,20):
        test_list=np.arange(lp,lp+20)
        for i in test_list: 
            if(i == age):
                tr=round(lp/20)  
                agecl=round(lp/20)    
    
    if ( bmi < 16):
        clbmi=4
    elif ( bmi >= 16 and bmi < 18.5):
        clbmi=3
    elif ( bmi >= 18.5 and bmi < 25):
        clbmi=2
    elif ( bmi >= 25 and bmi < 30):
        clbmi=1
    elif ( bmi >=30):
        clbmi=0    
    val1=DinnerfoodseparatedIDdata.describe()
    valTog=val1.T
    DinnerfoodseparatedIDdata=DinnerfoodseparatedIDdata.to_numpy()
    LunchfoodseparatedIDdata=LunchfoodseparatedIDdata.to_numpy()
    breakfastfoodseparatedIDdata=breakfastfoodseparatedIDdata.to_numpy()
    ti=(bmi+agecl)/2
    
    ## K-Means Based  Dinner Food
    Datacalorie=DinnerfoodseparatedIDdata[1:,1:len(DinnerfoodseparatedIDdata)]
    X = np.array(Datacalorie)
    kmeans = KMeans(n_clusters=3, random_state=0).fit(X)
    XValu=np.arange(0,len(kmeans.labels_))
    dnrlbl=kmeans.labels_
    
    ## K-Means Based  lunch Food
    Datacalorie=LunchfoodseparatedIDdata[1:,1:len(LunchfoodseparatedIDdata)]
    X = np.array(Datacalorie)
    kmeans = KMeans(n_clusters=3, random_state=0).fit(X)
    XValu=np.arange(0,len(kmeans.labels_))
    lnchlbl=kmeans.labels_
    
    ## K-Means Based  lunch Food
    Datacalorie=breakfastfoodseparatedIDdata[1:,1:len(breakfastfoodseparatedIDdata)]
    X = np.array(Datacalorie)
    kmeans = KMeans(n_clusters=3, random_state=0).fit(X)
    XValu=np.arange(0,len(kmeans.labels_))
    brklbl=kmeans.labels_
    inp=[]
    datafin=pd.read_csv('inputfin.csv')
    datafin.head(5)
    dataTog=datafin.T
    bmicls=[0,1,2,3,4]
    agecls=[0,1,2,3,4]
    weightlosscat = dataTog.iloc[[1,2,7,8]]
    weightlosscat=weightlosscat.T
    weightgaincat= dataTog.iloc[[0,1,2,3,4,7,9,10]]
    weightgaincat=weightgaincat.T
    healthycat = dataTog.iloc[[1,2,3,4,6,7,9]]
    healthycat=healthycat.T
    weightlosscatDdata=weightlosscat.to_numpy()
    weightgaincatDdata=weightgaincat.to_numpy()
    healthycatDdata=healthycat.to_numpy()
    weightlosscat=weightlosscatDdata[1:,0:len(weightlosscatDdata)]
    weightgaincat=weightgaincatDdata[1:,0:len(weightgaincatDdata)]
    healthycat=healthycatDdata[1:,0:len(healthycatDdata)]
    healthycatfin=np.zeros((len(healthycat)*5,9),dtype=np.float32)
    t=0
    r=0
    s=0
    yt=[]
    yr=[]
    ys=[]
    for zz in range(5):
        for jj in range(len(healthycat)):
            valloc=list(healthycat[jj])
            valloc.append(bmicls[zz])
            valloc.append(agecls[zz])
            healthycatfin[t]=np.array(valloc)
            yt.append(brklbl[jj])
            t+=1
        for jj in range(len(healthycat)):
            valloc=list(healthycat[jj])
            valloc.append(bmicls[zz])
            valloc.append(agecls[zz])
            healthycatfin[r]=np.array(valloc)
            yr.append(lnchlbl[jj])
            r+=1
        for jj in range(len(healthycat)):
            valloc=list(healthycat[jj])
            valloc.append(bmicls[zz])
            valloc.append(agecls[zz])
            healthycatfin[s]=np.array(valloc)
            ys.append(dnrlbl[jj])
            s+=1

    X_test=np.zeros((len(healthycat)*5,9),dtype=np.float32)
    for jj in range(len(healthycat)):
        valloc=list(healthycat[jj])
        valloc.append(agecl)
        valloc.append(clbmi)
        X_test[jj]=np.array(valloc)*ti   
    
    if res==1:
        X_train= healthycatfin
        y_train=yt
        
    elif res==2:
        X_train= healthycatfin
        y_train=yr
        
    elif res==3:
        X_train= healthycatfin
        y_train=ys
        
    
    from sklearn.ensemble import RandomForestClassifier
    
    clf=RandomForestClassifier(n_estimators=100)
    
    clf.fit(X_train,y_train)
    
    y_pred=clf.predict(X_test)

    if(res==1):
        num = 2;
    elif(res==2):
        num = 1;
    elif(res==3):
        num=0;
    return_list_data = [];
    for ii in range(len(Food_itemsdata)):
        if y_pred[ii]==num:
            return_list_data.append(Food_itemsdata[ii])
    return return_list_data;

# Get model
df = pkl.load(open('model.pkl','rb'));
fitness_mapping = {
    'Strength': ['Healthy'],
    'Plyometrics': ['Healthy'],
    'Cardio': ['Overweight', 'Underweight', 'Healthy'],
    'Stretching': ['Overweight', 'Underweight', 'Healthy'],
    'Powerlifting': ['Healthy'],
    'Strongman': ['Overweight'],
    'Olympic Weightlifting': ['Healthy']
}
def recommend_exercise(exer_type):
    exer= df.sort_values(by='Rating',ascending=False)
    exer_best =exer[exer['Type']==exer_type].head(5)
    return list(exer_best['Title'])

# Streamlit web app
st.title('Diet and Home Workout Recommendation System')
age_input = st.number_input(label='Age',value=18);
weight_input = st.number_input(label='Weight in (Kg)',value=60);
height_input = st.number_input(label='Height in (m)',value=1.75);
if st.button('Predict'):
    try:
        age = float(age_input);
        weight = float(weight_input);
        height = float(height_input);
        bmi = round(weight/(height**2),2);
        recommended_diet_plan = {}
        recommended_exercises = {}
        if ( bmi < 18.5):
            recommended_diet_plan['breakfast']=Weight_Gain(age,weight,height,1);
            recommended_diet_plan['lunch']=Weight_Gain(age,weight,height,2);
            recommended_diet_plan['dinner']=Weight_Gain(age,weight,height,3);
            for type in fitness_mapping.keys():
                if 'Underweight' in fitness_mapping[type]:
                    recommended_exercises[type] = recommend_exercise(type);
        elif ( bmi >= 18.5 and bmi < 25):
            recommended_diet_plan['breakfast']=Healthy(age,weight,height,1);
            recommended_diet_plan['lunch']=Healthy(age,weight,height,2);
            recommended_diet_plan['dinner']=Healthy(age,weight,height,3);
            for type in fitness_mapping.keys():
                if 'Healthy' in fitness_mapping[type]:
                    recommended_exercises[type] = recommend_exercise(type);
        elif ( bmi >= 25):
            recommended_diet_plan['breakfast']=Weight_Loss(age,weight,height,1);
            recommended_diet_plan['lunch']=Weight_Loss(age,weight,height,2);
            recommended_diet_plan['dinner']=Weight_Loss(age,weight,height,3);
            for type in fitness_mapping.keys():
                if 'Overweight' in fitness_mapping[type]:
                    recommended_exercises[type] = recommend_exercise(type);
        st.header('Recommended Diet Plan');
        for diet in recommended_diet_plan.keys():
            str = '<span style="color:crimson;font-size:1.2rem;font-weight:700">'+diet.capitalize()+'</span> : ';
            for i in range(len(recommended_diet_plan[diet])):
                str+= recommended_diet_plan[diet][i]
                if i!= len(recommended_diet_plan[diet]) -1:
                    str+=', '
            st.markdown(f"<p>{str}</p>", unsafe_allow_html=True)

        st.header('Recommended Exercise');
        for exercise in recommended_exercises.keys():
            str = '<span style="color:crimson;font-size:1.2rem;font-weight:700">'+exercise.capitalize()+'</span> : ';
            for i in range(len(recommended_exercises[exercise])):
                str+= recommended_exercises[exercise][i]
                if i!= len(recommended_exercises[exercise]) -1:
                    str+=', '
            st.markdown(f"<p>{str}</p>", unsafe_allow_html=True)

    except Exception as ex:
        st.warning(ex);