# -*- coding: utf-8 -*-
"""
Created on Tue Sep 23 15:09:32 2025

@author: Abdelwahab Bouzihay
"""
#import libraries
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, r2_score
import plotly.graph_objects as go
import plotly.express as px
import streamlit as st
from datetime import datetime
#define functions
def style_negative(v, props=''):
    '''style negative value in dataframe'''
    try:
        return props if v <0 else None
    except:
        pass
def style_positive(v, props=''):
    '''style  positive  value in dataframe'''
    try:
        return props if v>0 else None
    except:
        pass
def country_simple(country):
    ''' divide countries into 3 categories'''
    if country=='US':
        return 'USA'
    elif country=='IN':
        return 'India'
    else:
        return'Others'
def video_performance_analysis(df):
    ''' A  function that predict performace of features 
    perfirmance
    input: a dataframe 
    process:preprocess data, run RandomForestRegressor algorithm, generate
    prediction metrics
    return : modle '''
    df_clean = df.dropna()
    df_clean['Engagement_Rate']=(df_clean['Video Likes Added'] +
                     df_clean['User Comments Added'])/df_clean['Views']
    df_clean['Like_Ratio']=(df_clean['Video Likes Added']/
    df_clean['Video Likes Added'])
    features=['Video Length','Video Length','User Comments Added',
              'User Subscriptions Added']
    target ='Views'
    X= df_clean[features]
    y=df_clean[target]
    X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.33, random_state=42)
    model = RandomForestRegressor(n_estimators=100, random_state=43)
    model.fit(X_train,y_train)
    y_pred=model.predict(X_test)
    #Display results
    col1, col2=st.columns(2)
    with col1:
        st.metric('Model R² Score', f'{r2_score(y_test, y_pred):.3f}' )
        st.metric(' Mean absolute error',
                  f'{ mean_absolute_error(y_test, y_pred):.3f}')
    with col2:
        #features importance
       importanc_df= pd.DataFrame(
            {'features': features,
            'importance':model.feature_importances_}).sort_values('importance',
                                                    ascending=True )
    fig=px.bar(importanc_df, x='importance', y='features',
               title='Feature  Importance for View Prediction ')
    st.plotly_chart(fig, use_container_width=True )
    return model
        
#load dataand create cash data for Streamlit
@st.cache_data
def load_data():
    ''' Return data frames loaded with data fro, CSV fules'''
    
    df_agg = pd.read_csv('Aggregated_Metrics_By_Video.csv').iloc[1:,:]
    df_agg['Video pub­lish time']=pd.to_datetime(df_agg['Video pub­lish time'],
                                                 format='%b %d, %Y')
    df_agg['Av­er­age view dur­a­tion']=df_agg['Av­er­age view dur­a­tion'].apply(lambda x:
                                datetime.strptime(x,'%H:%M:%S'))
    df_agg['Av­er­age view dur­a­tion_sec']=df_agg['Av­er­age view dur­a­tion'].apply(lambda x:
                              x.second+ x.minute*60 + x.hour*3600)
    df_agg['engagement_ratio']=(df_agg['Com­ments ad­ded']+ df_agg['Shares']+ 
     df_agg['Likes']  + df_agg['Dis­likes'])/df_agg['Views']   
        
    df_agg['Views /gain'] = df_agg['Views']/df_agg['Sub­scribers gained']
    df_agg_subb=pd.read_csv('Aggregated_Metrics_By_Country_And_Subscriber_Status.csv')
    df_agg.sort_values('Video pub­lish time', ascending= False, inplace=True) 
    
    df_comments = pd.read_csv('Aggregated_Metrics_By_Video.csv')
    df_time= pd.read_csv('Video_Performance_Over_Time.csv')
    df_time['Date'] = pd.to_datetime(df_time['Date'],format='mixed')
    df_perf = pd.read_csv('Video_Performance_Over_Time.csv') 
    return df_agg, df_agg_subb,df_comments,df_time,df_perf

# call load_datato create data frames
df_agg, df_agg_subb,df_comments,df_time,df_perfor= load_data()
 

#create a copy of df_agg
df_agg_diff = df_agg.copy()
#replace hexadecimal soft hyphen with ''
df_agg_diff.columns = df_agg_diff.columns.str.replace('\xad', '')
#get the day 12 monts ince lasr day pyblished
metric_date_12mo =df_agg_diff['Video publish time'].max() - pd.DateOffset(months=12)
#select only numeric attributes
numeric_cols = df_agg_diff.select_dtypes(include=['int64', 'float64']).columns
#calculate median for 12 months since 
median_agg = df_agg_diff[df_agg_diff['Video publish time'] >=
                         metric_date_12mo][numeric_cols].median(numeric_only=True)

df_agg_diff.loc[:, numeric_cols]=(df_agg_diff.loc[:, numeric_cols]- median_agg).div(median_agg)
# build dashboard
add_side_barr =st.sidebar.selectbox('Aggregate or Individual Video',
                             ('Aggregate metrics', 'Video Performance prediction', 
                             'Individual Video Analysis'))

##Total picture
# What metrics will be relevant
## difference from baseline
## Percent change by vedioth. High percentage express the magnuted of change 
#from baseline.
if add_side_barr == 'Aggregate metrics':
    st.subheader("Aggregated data total picture")
    df_agg.columns = df_agg.columns.str.replace('\xad', '')
    df_agg_metrics= df_agg[[ 'Video publish time', 'Views', 'Likes', 
                'Subscribers', 'Shares', 'Comments added','RPM (USD)',
            'Average percentage viewed (%)', 'Average view duration_sec',
            'engagement_ratio', 'Views /gain' ]]
    metric_date_6mo = df_agg_metrics['Video publish time'].max() - pd.DateOffset(months=6)
    metric_date_12mo = df_agg_metrics['Video publish time'].max()- pd.DateOffset(months=12)

    metric_median6m=df_agg_metrics[ df_agg_metrics['Video publish time']>=
                                   metric_date_6mo ].median(numeric_only=True)
    metric_median12m=df_agg_metrics[df_agg_metrics['Video publish time']  
                         >=  metric_date_12mo ].median(numeric_only=True)
 
    col1,col2,col3, col4, col5 =st.columns(5)
    columns=[col1,col2,col3, col4, col5]
    count=0
    for i in metric_median6m.index:
        with columns[count]:
            delta = (metric_median6m[i]-metric_median12m[i])/metric_median12m[i]
            st.metric(i,round(metric_median6m[i],1),'{:.2%}'.format(delta))
            count +=1 
            if count==5:
                count=0
    df_agg_diff['Published date']=df_agg_diff['Video publish time'].apply(
                                                           lambda x: x.date())
    df_agg_diff_final =df_agg_diff.loc[:,['Video title','Published date',
            'Views', 'Likes', 'Subscribers','Average view duration_sec',
            'engagement_ratio','Views /gain'  ]]
    df_agg_numeric_lst = df_agg_diff_final.median(
        numeric_only=True).index.tolist()
    df_to_pct ={}
    for i in df_agg_numeric_lst:
        df_to_pct[i]='{:.1%}'.format
    
    st.dataframe( df_agg_diff_final.style.map(style_negative, props='color:red').format(df_to_pct))
    st.dataframe( df_agg_diff_final.style.map(style_positive, props='color:green').format(df_to_pct)) 
df_time_diff =pd.merge(df_time,   df_agg_diff.loc[:,
                ['Video','Video title','Video publish time']],
    left_on='External Video ID',right_on='Video') 

#Performance prediction
if add_side_barr == 'Video Performance prediction':
    # call performence model to get the future that can predict views
    model=video_performance_analysis(df_perfor)
#Individual video anlysiswill show how the subcribtions by 3 categories: USA,
#India and others displyed as a bar chart for selected video   
if add_side_barr == 'Individual Video Analysis':
    st.subheader("Individual Video performance")
    videos = tuple(list(df_agg_diff['Video title']))
 
    video_select= st.selectbox('Pick a video', videos)

    agg_filtered= df_agg_diff[df_agg_diff['Video title']== video_select]
    agg_sub_filtered= df_agg_subb[df_agg_subb['Video Title']== video_select]
    agg_sub_filtered['Country']=agg_sub_filtered['Country Code'].apply(country_simple)
    agg_sub_filtered.sort_values('Is Subscribed', inplace =True)
    st.subheader("Individual Video Subscription per category chart")
    fig = px.bar(agg_sub_filtered, x='Views', y='Is Subscribed',
                  color ='Country', orientation= 'h')
    st.plotly_chart(fig) 
    
#merge daily data with dily data to get delata time
   
    df_time_diff['days_published']=(df_time_diff['Date'] -
                                    df_time_diff[ 'Video publish time']).dt.days
    #get 12 month data from  all data
    date_12m= df_agg_diff[ 'Video publish time'].max() - pd.DateOffset(months=12) 
    df_time_diff_yr=df_time_diff[df_time_diff[ 'Video publish time']>date_12m]
    #get daily view data (30 days) median & percentil
    view_days =pd.pivot_table(df_time_diff_yr, index='days_published',values='Views',
       aggfunc=[np.mean,np.median,  lambda x: np.percentile(x,80),
                lambda x: np.percentile(x,20)]).reset_index()
    view_days.columns=['days_published','mean_views', 'median_views',
                       '80pct_views', '20pct_views']
    view_days=view_days[view_days['days_published'].between(0,30)]
    views_cummulative=view_days.loc[:,['days_published','mean_views',
                                     'median_views','80pct_views', '20pct_views']]
    views_cummulative.loc[:,['mean_views','median_views',
     '80pct_views', '20pct_views']]=views_cummulative.loc[:,['mean_views',
            'median_views','80pct_views', '20pct_views']].cumsum()
    ##
    
    agg_time_filtered= df_time_diff[df_time_diff['Video Title']== video_select]
    first_30days=agg_time_filtered[agg_time_filtered['days_published'].between(0,30)]
    first_30days=first_30days.sort_values('days_published')
    fig2 = go.Figure()
    fig2.add_trace(go.Scatter(x=views_cummulative['days_published'],
        y=views_cummulative['20pct_views'] ,mode='lines',
        name='20th percentile' , line=dict(color='purple',dash='dash')))
    fig2.add_trace(go.Scatter(x=views_cummulative['days_published'],
        y=views_cummulative['80pct_views'] ,mode='lines',
        name='20th percentile' , line=dict(color='black',dash='dash')))
    fig2.add_trace(go.Scatter(x=views_cummulative['days_published'],
        y=views_cummulative['median_views'] ,mode='lines + markers',
        name='50th percentile' , line=dict(color='royalblue',dash='dash')))
    fig2.add_trace(go.Scatter(x= first_30days['days_published'],
        y=first_30days['Views'].cumsum() ,mode='lines + markers',
        name='Cuurent Videos' , line=dict(color='firebrick',width=8)))
    fig2.update_layout(title='Viwes comparaison first 30 days',
                       xaxis_title='Days since published',
                       yaxis_title='Cummulative views')
    st.plotly_chart(fig2)
# What metrics will be relevant
## difference from baseline
## Percent change by vedio


##Individual Vedio


#Improvement
