import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import re
from sklearn.model_selection import StratifiedKFold
from sklearn.model_selection import KFold
from sklearn.linear_model import LinearRegression
from sklearn.linear_model import ElasticNetCV
from sklearn.linear_model import ElasticNet
from sklearn import preprocessing
from sklearn.linear_model import Ridge
from sklearn.linear_model import Lasso
from sklearn.neural_network import MLPRegressor
from sklearn.datasets import make_regression
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.datasets import make_classification
from sklearn.neighbors import KNeighborsRegressor
from sklearn.metrics import mean_squared_error
from sklearn.metrics import mean_absolute_error
from sklearn.dummy import DummyRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.svm import SVR
from sklearn.ensemble import ExtraTreesRegressor
from sklearn.ensemble import AdaBoostRegressor
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.ensemble import VotingRegressor
import random
from sklearn.inspection import permutation_importance
from operator import itemgetter
import time


#take a correlation matrix and a the matrix dataframe and show if the correlation came from blank space or not
def good_or_bad_correlation(correlation_matrix,dataframe_correlate):
    correlation_columns = []
    for num,column in enumerate(correlation_matrix.columns):
        for row1 in correlation_matrix.columns[num+1:]:
            comptnan = 0
            comptvalues  = 0
            if correlation_matrix[column][row1] >= 0.99:
                print("-------------------------------------------------------------------------------")
                if ((dataframe_correlate[column].notna()) & (dataframe_correlate[row1].notna())).sum() > 25:
                    display(dataframe_correlate[((dataframe_correlate[column].notna()) & (dataframe_correlate[row1].notna()))][[column,row1]].head(5))
                    display(dataframe_correlate[((dataframe_correlate[column].isna()) & (dataframe_correlate[row1].notna()))][[column,row1]].head(5))
                    display(dataframe_correlate[((dataframe_correlate[column].notna()) & (dataframe_correlate[row1].isna()))][[column,row1]].head(5))
                    correlation_columns += [[column,row1]]
                else:
                    print("Fausse correlation de 1 pour ",column," and ",row1) 
    return(correlation_columns)

# plot cross validation for 5 k fold
def cross_validation_scores(y_values):
    sns.set(style="white", rc={"lines.linewidth": 3})
    sns.barplot(x=['Iter1','Iter2','Iter3','Iter4','Iter5'],y=y_values)
    plt.show()

# plot histogram off two score from two dictionnary
def plot_result_vs_expected(expected,result):
    result_combine = pd.DataFrame()
    result_combine["expected"] = expected
    result_combine["result"] = result
    result_combine.sort_values(by=["expected"],ignore_index=True,inplace = True)                       
    fig = plt.figure()
    ax1 = fig.add_subplot(111)
    ax1.scatter(result_combine.index, result_combine["expected"], s=10, c='b', marker="s", label='expected')
    ax1.scatter(result_combine.index,result_combine["result"], s=10, c='r', marker="o", label='result')
    plt.legend(loc='upper left');
    plt.show()

    
# boxplot 
def boxplot_qcut(x_value,y_value,title):
    sns.set_theme(style="ticks", palette="pastel")
    ax = sns.boxplot(x=x_value,
                    y=y_value)
    sns.despine(offset=10, trim=True)
    smt = plt.setp(ax.get_xticklabels(), rotation=90)
    ax.set_title(title)

    
    
# take targets and separate it homogeneously in k part with a number_of_parts accuracy
# return index list separated
def stratified_k_fold_based_on_y(y,k,number_of_parts):
    first_quantile = 1 / number_of_parts
    interval_max = first_quantile
    interval_min = y.min()
    total_list_of_index_separated = [ [] for _ in range(number_of_parts) ]
    index_list_for_stratified_k_fold = [ [[],[]] for _ in range(k)]
    valuestest = [ [] for _ in range(number_of_parts) ]
    for part_number in range(number_of_parts):
        if part_number == 0:
            separator_value = y.quantile(interval_max)
            total_list_of_index_separated[part_number] = y[y < separator_value].index.tolist()
            valuestest[part_number] = y[y < separator_value].values.tolist()
            interval_min = interval_max
            interval_max += first_quantile
        elif part_number > 0 and part_number < number_of_parts - 1:
            separator_value2 = separator_value
            separator_value = y.quantile(interval_max)
            total_list_of_index_separated[part_number] = y[(y < separator_value) & (y > separator_value2)].index.tolist()
            valuestest[part_number] = y[(y < separator_value) & (y > separator_value2)].values.tolist()
            interval_min = interval_max
            interval_max += first_quantile
        else:
            total_list_of_index_separated[part_number] = y[y > separator_value].index.tolist()
            valuestest[part_number] = y[y > separator_value].values.tolist()    
    for part_number in range(number_of_parts):
        random.Random(14).shuffle(total_list_of_index_separated[part_number])
    
    
    for k_index in range(k):
        
        for part_number in range(number_of_parts):
            interval_index = int(len(total_list_of_index_separated[part_number]) / k)
            if k_index == 0:
                index_max = interval_index
                for i,value in enumerate(total_list_of_index_separated[part_number]):
                    if  i < index_max:
                        index_list_for_stratified_k_fold[k_index][1].append(value)
                    else:
                        index_list_for_stratified_k_fold[k_index][0].append(value)
            elif k_index > 0 and k_index < k - 1:
                index_max = interval_index * (k_index + 1)
                index_min = interval_index * k_index
                for i,value in enumerate(total_list_of_index_separated[part_number]):
                    if i >= index_min and i < index_max:
                        index_list_for_stratified_k_fold[k_index][1].append(value)
                    else:
                        index_list_for_stratified_k_fold[k_index][0].append(value)
            else:
                index_min = interval_index * k_index
                for i,value in enumerate(total_list_of_index_separated[part_number]):
                    if i >= index_min:
                        index_list_for_stratified_k_fold[k_index][1].append(value)
                    else:
                        index_list_for_stratified_k_fold[k_index][0].append(value)
    return index_list_for_stratified_k_fold

# plot top_features most importante algorithm_fitted features
# return the all list off features with importances score list
# mean decrease impurity
def plot_variable_importance_mdi(algorithm_fitted,feature_names,top_features):
    importances = algorithm_fitted.feature_importances_
    std = np.std([tree.feature_importances_ for tree in algorithm_fitted.estimators_], axis=0)
    forest_importances = pd.Series(importances, index=feature_names)
    forest_importances_return = forest_importances.copy()
    top_importance = pd.Series(dtype='float64')
    for feature in range(top_features):
        argmax = forest_importances.argmax()
        index = forest_importances.index[argmax]
        top_importance[index] = forest_importances[argmax]
        forest_importances = forest_importances.drop(labels=index)
    
    
    fig, ax = plt.subplots()
    top_importance.plot.bar(ax=ax)##yerr=top_importance, ax=ax)
    ax.set_title("Feature importances using mean decrease in impurity")
    ax.set_ylabel("Mean decrease in impurity")
#     print(forest_importances_return.sort_values(ascending=False).head(10))  
    return forest_importances_return

# plot top_features most importante algorithm_fitted features
# return the all list off features with importances score list
# permutation features measurement
def plot_variable_importance_pfm(algorithm_fitted,X_test,y_test,feature_names,top_features):
    result = permutation_importance(algorithm_fitted, X_test, y_test, n_repeats=10, random_state=42, n_jobs=2)
    forest_importances = pd.Series(result.importances_mean, index=feature_names)
#     print(max(result.importances_mean),max(result.importances_std))
    top_importance = pd.Series(dtype='float64')
    forest_importances_return = forest_importances.copy()
    for feature in range(top_features):
        argmax = forest_importances.argmax()
        index = forest_importances.index[argmax]
        top_importance[index] = forest_importances[argmax]
        forest_importances = forest_importances.drop(labels=index)
    fig, ax = plt.subplots()
#     forest_importances.plot.bar(yerr=result.importances_std, ax=ax)
    top_importance.plot.bar(ax=ax)##yerr=top_importance, ax=ax)
    ax.set_title("Feature importances using permutation on full model")
    ax.set_ylabel("Mean accuracy decrease")
    plt.show()
#     print(forest_importances_return.sort_values(ascending=False).head(10))  
    return forest_importances_return

# histplot
def hisplot_y_part_count(y,bins_number,title):
    fig = sns.histplot(
        y,
        multiple="stack",
        palette="light:m_r",
        edgecolor=".3",
        bins=bins_number
    )
    fig.set_title(title)

# take (features, target, a boolea to say if y is log or not, the k fold index list, use of standard scale, the algorythm used, the parametre for the algorithm)
# plot target values, graph for each k fold
# calcul and show r squarred score
# calcul and show mae and rmse score and return it
def r2_skf_regression(X,y,not_log_result,k_fold_index,standard_scale,algorythm,**args):
    r2_model = []
    rmse = []
    mae = []
    for train_index, test_index in k_fold_index:
        X_train, X_test = X[train_index], X[test_index]
        y_train, y_test = y[train_index], y[test_index]
        std_scale = preprocessing.StandardScaler().fit(X_train)
        if standard_scale:
            X_train_std = std_scale.transform(X_train)
            X_test_std = std_scale.transform(X_test)
            model_skf = algorythm(**args).fit(X_train_std, y_train)
            r2_model.append(model_skf.score(X_test_std, y_test)*100)
            if not_log_result:
                rmse.append(np.sqrt(mean_squared_error(10 ** y_test,10 ** model_skf.predict(X_test_std))))
                mae.append(mean_absolute_error(10 ** y_test,10 ** model_skf.predict(X_test_std)))
            else:            
                rmse.append(np.sqrt(mean_squared_error(y_test,model_skf.predict(X_test_std))))
                mae.append(mean_absolute_error(y_test,model_skf.predict(X_test_std)))
            plot_result_vs_expected(y_test,model_skf.predict(X_test_std))
        else:
            model_skf = algorythm(**args).fit(X_train, y_train)
            r2_model.append(model_skf.score(X_test, y_test)*100)
            if not_log_result:
                rmse.append(mean_squared_error(10 ** y_test,10 ** model_skf.predict(X_test)))
                mae.append(mean_absolute_error(10 ** y_test,10 ** model_skf.predict(X_test)))
            else:
                rmse.append(mean_squared_error(y_test,model_skf.predict(X_test)))
                mae.append(mean_absolute_error(y_test,model_skf.predict(X_test)))
            plot_result_vs_expected(y_test,model_skf.predict(X_test))
    
    print("r2",r2_model)
    if not_log_result:
        print("mean",(10 ** y).mean())
    else:
        print("mean",y.mean())
    print("rmse",rmse)
    print("mae",mae)
    
    return rmse,mae

# calcul variables importance with pfm or mdi algorythm and return the score of all variables
def variable_importance(X,y,k_fold_index,standard_scale,mdi_or_pfm,top_features_quantity,X_names,algorythm,**args):
    std_scale = preprocessing.StandardScaler().fit(X)
    variable_importance_list = []
    for train_index, test_index in k_fold_index:
        X_train, X_test = X[train_index], X[test_index]
        y_train, y_test = y[train_index], y[test_index]
        if standard_scale:
            X_train_std = std_scale.transform(X_train)
            X_test_std = std_scale.transform(X_test)
            model_skf = algorythm(**args).fit(X_train_std, y_train)
        else:
            model_skf = algorythm(**args).fit(X_train, y_train)
        if mdi_or_pfm == "mdi":
            variable_importance_list.append(plot_variable_importance_mdi(model_skf,X_names,top_features_quantity))
        elif mdi_or_pfm == "pfm":
            variable_importance_list.append(plot_variable_importance_pfm(model_skf,X_test,y_test,X_names,top_features_quantity))
        else:
            print("mdi_or_pfm error value")
          
    return(variable_importance_list)

# show graph with algorithmes comparaison
def print_dictionnary_head_score(dictionnary,head_top,title):
    columns=["value1","value2","value3","value4","value5"]
    result_dataframe = pd.DataFrame(columns=columns)
    for key, value in dictionnary.items():
        result_dataframe.loc[key] = value
#     for column in columns:
#         print(result_dataframe.sort_values(by=column,ascending=False).head(head_top)[column],"\n\n")
    result_dataframe2 = result_dataframe.rdiv(result_dataframe.min()).copy()
    result_dataframe["mean"] = 0
    result_dataframe = result_dataframe.cumsum(axis = 1)
    result_dataframe["mean"] = result_dataframe["mean"].div(5)
    

    result_dataframe2["mean_score"] = 0
    result_dataframe2 = result_dataframe2.cumsum(axis = 1)
    result_dataframe2["mean_score"] = result_dataframe2["mean_score"].div(5)
    result_dataframe["mean_score"] = result_dataframe2["mean_score"]
    sns.set(style="white", rc={"lines.linewidth": 3})
    
    ax = sns.barplot(x=result_dataframe2.sort_values(by="mean_score",ascending=False).head(head_top)["mean_score"].index,
                     y=result_dataframe2.sort_values(by="mean_score",ascending=False).head(head_top)["mean_score"])
    smt = plt.setp(ax.get_xticklabels(), rotation=90)
    for i, value in enumerate(result_dataframe.sort_values(by="mean_score",ascending=False).head(head_top)["mean"]):
        ax.text(i - 0.1 , 0.4, str(round(value,4)), color='black',rotation = "vertical")
    ax.set_title(title)
    plt.show()

# show the variables classement top list and return    
def print_variable_classement_from_dictionnary(top_variable_list):
    first_array = top_variable_list[0][0]
    column_value = "value1"
    iterator = 1
    variable_score_dataframe = pd.DataFrame(index=first_array.index)
    
    for sublist in top_variable_list:
        for subsublist in sublist:
            variable_score_dataframe["value"+str(iterator)] = subsublist
            iterator += 1
    column_list = variable_score_dataframe.columns
    column_count = len(column_list)
    semi_column_count = column_count / 2
    variable_score_dataframe["intersection"] = (variable_score_dataframe[variable_score_dataframe[column_list[:int(semi_column_count)]] != 0].any(1) & variable_score_dataframe[variable_score_dataframe[column_list[int(semi_column_count):]] != 0].any(1))
    display(variable_score_dataframe[variable_score_dataframe["intersection"]==True].index)
    return(variable_score_dataframe[variable_score_dataframe["intersection"]==True].index)

# plot the two dictionnary to compare algorithm score
def gain_or_loose_plot(dictionnary1,dictionnary2,title):
    dict1 = dictionnary1.copy()
    dict2 = dictionnary2.copy()
    result_df = pd.DataFrame(columns=["with_or_without","score_label","value"])
    for i, sublist in enumerate(dict2):
        mean2 = np.mean(dict2[sublist])
        mean1 = np.mean(dict1[sublist])
        difference = mean2 - mean1
        result_df = result_df.append(pd.DataFrame([["mean_without_energyStarScore",sublist,round(mean1,3)]],columns=["with_or_without","score_label","value"]),ignore_index=True)
        result_df = result_df.append(pd.DataFrame([["mean_with_energyStarScore",sublist,round(mean2,3)]],columns=["with_or_without","score_label","value"]),ignore_index=True)
    
    sns.set_theme(style="whitegrid")
    g = sns.catplot(
        data=result_df, kind="bar",
        x="score_label", y="value", hue="with_or_without",
        ci="sd",palette="dark",
        alpha=.6, height=6)
    g.set_xticklabels(rotation=90)
    g.despine(left=True)
    g.fig.subplots_adjust(left=0.3) # adjust the Figure in rp
    g.fig.suptitle(title)
    g.set_axis_labels("", "score")
    g.legend.set_title("")

# algorythm to mix two machine learning algorithmes    
def voting_regressor_two_mix(X,y,not_log_result,k_fold_index,standard_scale,regression1,regression2):
    #     skf = StratifiedKFold()#shuffle=True, random_state=1)
    r2_model = []
    rmse = []
    mae = []
    std_scale = preprocessing.StandardScaler().fit(X)
    er = VotingRegressor([('reg1', regression1), ('reg2', regression2)])
    for train_index, test_index in k_fold_index:
        X_train, X_test = X[train_index], X[test_index]
        y_train, y_test = y[train_index], y[test_index]
        if standard_scale:
            X_train_std = std_scale.transform(X_train)
            X_test_std = std_scale.transform(X_test)
            model_skf = er.fit(X_train_std, y_train)
            r2_model.append(model_skf.score(X_test_std, y_test)*100)
            if not_log_result:
                rmse.append(mean_squared_error(10 ** y_test,10 ** model_skf.predict(X_test_std)))
                mae.append(mean_absolute_error(10 ** y_test,10 ** model_skf.predict(X_test_std)))
            else:            
                rmse.append(mean_squared_error(y_test,model_skf.predict(X_test_std)))
                mae.append(mean_absolute_error(y_test,model_skf.predict(X_test_std)))
            plot_result_vs_expected(y_test,model_skf.predict(X_test_std))
        else:
            model_skf = er.fit(X_train, y_train)
            r2_model.append(model_skf.score(X_test, y_test)*100)
            if not_log_result:
                rmse.append(mean_squared_error(10 ** y_test,10 ** model_skf.predict(X_test)))
                mae.append(mean_absolute_error(10 ** y_test,10 ** model_skf.predict(X_test)))
            else:
                rmse.append(mean_squared_error(y_test,model_skf.predict(X_test)))
                mae.append(mean_absolute_error(y_test,model_skf.predict(X_test)))
            plot_result_vs_expected(y_test,model_skf.predict(X_test_std))
    print("r2",r2_model)
    if not_log_result:
        print("mean",(10 ** y).mean())
    else:
        print("mean",y.mean())
    print("rmse",rmse)
    print("mae",mae)
    return rmse,mae

# start of chronometer
def start():
    start = time.time()
    return(start)

# end and print of the execution time
def stop(start):
    end = time.time()
    elapsed = end - start
    print(f'Temps d\'exécution : {elapsed:.2}s')
    
#  score print from gridsearchCV
def print_score(clf):
    score = 'mae'
    for mean, std, params in zip(
            clf.cv_results_['mean_test_score'], # score moyen
            clf.cv_results_['std_test_score'],  # écart-type du score
            clf.cv_results_['params']           # valeur de l'hyperparamètre
        ):

        print("{} = {:.3f} (+/-{:.03f}) for {}".format(
            score,
            mean,
            std*2,
            params
        ) )
    print("\n\n")