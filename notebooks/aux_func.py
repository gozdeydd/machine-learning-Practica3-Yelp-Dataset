import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import scipy.stats as ss
from sklearn import metrics
from sklearn.metrics import roc_auc_score, accuracy_score, classification_report, confusion_matrix, roc_curve
import pickle
from sklearn.metrics import ConfusionMatrixDisplay
import re
import matplotlib


def duplicate_columns(frame):
    groups = frame.columns.to_series().groupby(frame.dtypes).groups
    dups = []

    for t, v in groups.items():

        cs = frame[v].columns
        vs = frame[v]
        lcs = len(cs)

        for i in range(lcs):
            ia = vs.iloc[:,i].values
            for j in range(i+1, lcs):
                ja = vs.iloc[:,j].values
                if np.array_equal(ia, ja):
                    dups.append(cs[i])
                    break
    return dups




### Función Métricas del Modelo
def evaluate_model(ytest, ypred, ypred_proba = None):
    if ypred_proba is not None:
        print('ROC-AUC score of the model: {}'.format(roc_auc_score(ytest, ypred_proba[:, 1])))
    print('Accuracy of the model: {}\n'.format(accuracy_score(ytest, ypred)))
    print('Classification report: \n{}\n'.format(classification_report(ytest, ypred)))
    print('Confusion matrix: \n{}\n'.format(confusion_matrix(ytest, ypred)))

### Función para cargar el Modelo
def cargar_modelo(ruta):
    return pickle.load(open(ruta, 'rb'))


### Función agregadora de Métricas, Matriz de confusión, curva ROC y threshold óptimo
def model_analysis(modelo, xtest, ytest):
    matplotlib.rcParams['figure.figsize'] = (9, 9)
    ypred = modelo.predict(xtest)
    ypred_proba = modelo.predict_proba(xtest)
    # keep probabilities for the positive outcome only
    yhat = ypred_proba[:, 1]
    # calculate roc curves
    fpr, tpr, thresholds = roc_curve(ytest, yhat)
    # plot the roc curve for the model
    plt.plot([0, 1], [0, 1], linestyle='--', label='No Skill')
    plt.plot(fpr, tpr, marker='.', label=re.findall('^[A-z]+', str(modelo)))
    # axis labels
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.legend()
    # show the plot
    plt.show()

    gmeans = np.sqrt(tpr * (1 - fpr))
    # locate the index of the largest g-mean
    ix = np.argmax(gmeans)

    # print('Best Threshold=%f, G-Mean=%.3f' % (thresholds[ix], gmeans[ix]))

    # plot the roc curve for the model
    plt.plot([0, 1], [0, 1], linestyle='--', label='No Skill')
    plt.plot(fpr, tpr, marker='.', label=re.findall('^[A-z]+', str(modelo)))
    plt.scatter(fpr[ix], tpr[ix], s=100, marker='o', color='black', label='Best')
    # axis labels
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.legend()
    # show the plot
    plt.show()

    ypred_new_threshold = (ypred_proba[:, 1] > thresholds[ix]).astype(int)

    # Plot non-normalized confusion matrix
    titles_options = [("Confusion matrix, without normalization", None),
                      ("Normalized confusion matrix", 'true')]

    for title, normalize in titles_options:
        fig, ax = plt.subplots(figsize=(10, 10))
        disp = ConfusionMatrixDisplay.from_predictions(ytest, ypred_new_threshold,
                                                       cmap=plt.cm.Greens,
                                                       normalize=normalize,
                                                       ax=ax)
        ax.set_title(title)

    evaluate_model(ytest, ypred_new_threshold, ypred_proba)
    
    
# Histogram with Kernel Density Estimate Plot (KDE)
# Source: https://stackoverflow.com/questions/43638851/pandas-histogram-plot-with-kde
def plot_histograms(df, columns):
    # keep total number of subplot
    k = len(df.columns)
    # n = number of chart columns
    n = columns
    m = (k - 1) // n + 1
    
    # Create figure
    fig, axes = plt.subplots(m, n, figsize=(n * 5, m * 3))

    # Iterate through columns, tracking the column name and 
    # which number we are at i. Within each iteration, plot
    for i, (name, col) in enumerate(df.iteritems()):
        r, c = i // n, i % n
        ax = axes[r, c]
        # the histogram
        col.hist(ax=ax)
        
        # kde = Kernel Density Estimate plot
        ax2 = col.plot.kde(ax=ax, secondary_y=True, title=name)
        ax2.set_ylim(0)
        
        

    # Use tight_layout() as an easy way to sharpen up the layout spacing
    fig.tight_layout()


def get_feature_names(column_transformer):
    """Get feature names from all transformers.
    Returns
    -------
    feature_names : list of strings
        Names of the features produced by transform.
    """
    # Remove the internal helper function
    #check_is_fitted(column_transformer)
    
    # Turn loopkup into function for better handling with pipeline later
    def get_names(trans):
        # >> Original get_feature_names() method
        if trans == 'drop' or (
                hasattr(column, '__len__') and not len(column)):
            return []
        if trans == 'passthrough':
            if hasattr(column_transformer, '_df_columns'):
                if ((not isinstance(column, slice))
                        and all(isinstance(col, str) for col in column)):
                    return column
                else:
                    return column_transformer._df_columns[column]
            else:
                indices = np.arange(column_transformer._n_features)
                return ['x%d' % i for i in indices[column]]
        if not hasattr(trans, 'get_feature_names'):
        # >>> Change: Return input column names if no method avaiable
            # Turn error into a warning
            warnings.warn("Transformer %s (type %s) does not "
                                 "provide get_feature_names. "
                                 "Will return input column names if available"
                                 % (str(name), type(trans).__name__))
            # For transformers without a get_features_names method, use the input
            # names to the column transformer
            if column is None:
                return []
            else:
                return [name + "__" + f for f in column]

        return [name + "__" + f for f in trans.get_feature_names()]
    
    ### Start of processing
    feature_names = []
    
    # Allow transformers to be pipelines. Pipeline steps are named differently, so preprocessing is needed
    if type(column_transformer) == sklearn.pipeline.Pipeline:
        l_transformers = [(name, trans, None, None) for step, name, trans in column_transformer._iter()]
    else:
        # For column transformers, follow the original method
        l_transformers = list(column_transformer._iter(fitted=True))
    
    
    for name, trans, column, _ in l_transformers: 
        if type(trans) == sklearn.pipeline.Pipeline:
            # Recursive call on pipeline
            _names = get_feature_names(trans)
            # if pipeline has no transformer that returns names
            if len(_names)==0:
                _names = [name + "__" + f for f in column]
            feature_names.extend(_names)
        else:
            feature_names.extend(get_names(trans))
    
    return feature_names

# We define a function that would replace True and Falses with 1 and 0
def process_columns(df, columns):
    """Replace null values with 0 and change the data type to int for the specified columns."""
    for col in columns:
        df[col] = df[col].fillna(0)
        df[col] = df[col].replace({'True' : True, 'False': False, 0 : False})        
        df[col] = df[col].astype(bool)
    return df
# quickly check data info after reading it, or changing something.
def preview_data(df):
    print("Dataset has ", len(df.index), "rows and ", len(df.columns), "columns")
    return df.head()

#We create an information dataframe for business dataset
def check_missing_feautre(df):
    print('Missing values and datatypes of dataframe')
    df_dtypes = pd.merge(df.isnull().sum(axis = 0).sort_values().to_frame('missing_value').reset_index(),
         df.dtypes.to_frame('feature_type').reset_index(),
         on = 'index',
         how = 'inner')
    return(df_dtypes)

#function to load json chunsks and merge them
def load_rows(filepath, skip=0, nrows = None):
    with open(filepath, encoding='utf8') as json_file:
        read_count, load_count = 0, 0
        objs = []
        line = json_file.readline()
        while (nrows is None or load_count < nrows) and line:
            read_count += 1
            if read_count > skip:
                obj = json.loads(line)
                objs.append(obj)
                load_count += 1
                if load_count % 10000 == 0:
                    print(load_count, 'loaded')
            line = json_file.readline()
    return pd.DataFrame(objs)