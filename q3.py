import pandas as pd
import math
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
import pptree
import scipy.stats as stats


class Node:
    def __init__(self, parent=None, df=None, split_value="", split_by="", target='', leaf=False, children=None):
        self.parent = parent
        self.df = df
        self.split_value = split_value
        self.split_by = split_by
        self.target = target
        self.leaf = leaf
        self.children = []

    def __str__(self):
        if self.split_value != '' and self.target == '':
            return self.split_by + " = " + self.split_value
        elif self.split_value != '' and self.target != '':
            if self.target == 1:
                prediction = 'busy'
            else:
                prediction = 'not busy'
            return self.split_by + " = " + self.split_value + " -->>" + prediction
        else:
            return "Start"


def fix_features_types(df):
    """
    :param df: df
    :return: df with fixed feature types.
    """
    df.Date = pd.to_datetime(df.Date)
    df.Seasons = df.Seasons.astype("string")

    return df


def feature_extraction(df):
    """
    :param df: df
    :return: df with new features.
    """
    df.Holiday = df.Holiday.astype("string").apply(lambda x: True if x == "Holiday" else False if x == "No Holiday" else 0)
    df["Functioning Day"] = df["Functioning Day"].astype("string").apply(lambda x: True if x == "Yes" else False if x == "No" else 0)
    df['day'] = df.Date.apply(lambda x: x.weekday())
    df['month'] = df['Date'].apply(lambda x: x.month)
    df['busy'] = df['Rented Bike Count'].apply(lambda x: 1 if x > 650 else 0)
    return df


def print_corr_matrix(df):
    """
    :param df: df.
    :return: print the correlation matrix, for all numeric features.
    """
    numerics = ['int16', 'int32', 'int64', 'float16', 'float32', 'float64']
    corr = df.select_dtypes(include=numerics).corr()
    corr.style.background_gradient(cmap='coolwarm')
    sns.heatmap(corr, annot=True)
    plt.show()


def feature_selection(df):
    # print_corr_matrix(df) # the def I built calculate the correlation matrix
    # drop the 'Dew point temperature(°C)' because it has 91% correlation with the temperature therefore redundant. when I didnt drop it, the accuracy was 65%+- instead of 80%+ without it.
    # drop the date because it has too many different values, I extracted the important parts from it (day of the week, month)
    return df.drop(columns=['Date', 'Rented Bike Count', 'Dew point temperature(°C)'])


def feature_representation(df):
    """
    :param df: df.
    :return: df after Discretization.
    """
    df['day'] = df['day'].apply(lambda x: "Monday to Thursday" if x in [0, 1, 2, 3] else "Friday to Sunday")
    # round the hour to 3 parts of the day
    df['Hour'] = df['Hour'].apply(lambda x: '00:00-08:00' if x < 8 else '08:00-16:00' if x < 16 else '16:00-00:00')
    # round the temp to multiples of 5
    df['Temperature(°C)'] = df['Temperature(°C)'].apply(lambda x: x - (x % 10))
    df['Temperature(°C)'] = df['Temperature(°C)'].apply(lambda x: f"{x} to {x+10}(°C)")
    # change to one range all Temperatures lower than 0
    df['Temperature(°C)'] = df['Temperature(°C)'].apply(lambda x: "less than 0(°C)" if x[0] == '-' else x)
    # round down Humidity(%) to multiples of 20
    df['Humidity(%)'] = df['Humidity(%)'].apply(lambda x: x - (x % 40))
    df['Humidity(%)'] = df['Humidity(%)'].apply(lambda x: f"{x}-{x + 40}(%)")
    # round down Humidity(%) to multiples of 20
    df['Wind speed (m/s)'] = df['Wind speed (m/s)'].apply(lambda x: x - (x % 2))
    df['Wind speed (m/s)'] = df['Wind speed (m/s)'].apply(lambda x: f"{x} to {x + 2}(m/s)")
    # round down Visibility (10m) to multiples of 500
    df['Visibility (10m)'] = df['Visibility (10m)'].apply(lambda x: x - (x % 1500))
    df['Visibility (10m)'] = df['Visibility (10m)'].apply(lambda x: f"{x}-{x+1500}")
    # round Solar Radiation (MJ/m2)
    df['Solar Radiation (MJ/m2)'] = df['Solar Radiation (MJ/m2)'].apply(round)
    df['Solar Radiation (MJ/m2)'] = df['Solar Radiation (MJ/m2)'].apply(lambda x: f"{x} - {x+1}")
    # Rainfall(mm)
    df['Rainfall(mm)'] = df['Rainfall(mm)'].apply(lambda x: "0(mm)" if x == 0 else "0 to 5(mm)" if x <= 5 else "more than 5(mm)")
    # Snowfall (cm)
    df['Snowfall (cm)'] = df['Snowfall (cm)'].apply(lambda x: "No Snow" if x == 0 else "Snow")
    df['month'] = df['month'].apply(lambda x: x - (x % 2))
    df['month'] = df['month'].apply(lambda x: f"{x} to {x + 2}")
    return df


def majority_vote(df):
    """
    :param df: df
    :return: the majority value in the target ('busy') column.
    """
    num_of_busy = len(df.loc[df.busy == 1])
    if num_of_busy == len(df) / 2:
        return 1 if np.random.rand() > 0.5 else 0        # random choice for cases the are equal amount of True and False in the target feature
    else:
        return 1 if num_of_busy / len(df) > 0.5 else 0


def calculate_entropy(df, col_name):
    """
    :param df: df
    :param col_name: name of one of the columns from the df.
    :return: entropy.
    """
    entropy = 0
    for value in df[col_name].drop_duplicates():
        proportion = len(df.loc[df[col_name] == value]) / len(df)
        entropy += -1 * proportion * math.log2(proportion)
    return entropy


def get_max_entropy_feature(df):
    """
    :param df: df.
    :return: the column with the highest entropy.
    """
    max_information_gain = 0
    features = list(df.columns)
    features.remove('busy')
    max_information_gain_feature = features[0]
    for feature in features:
        information_gain = calculate_entropy(df, feature)
        if information_gain > max_information_gain:
            information_gain = max_information_gain
            max_information_gain_feature = feature
    return max_information_gain_feature


def get_the_ready_df(df=pd.read_csv(f"SeoulBikeData.csv", encoding='unicode_escape')):
    """
    :param df: df
    :return: df after all the important pre processing.
    """
    df = fix_features_types(df=df)
    df = feature_extraction(df=df)
    df = feature_selection(df=df)
    return feature_representation(df=df)


def recursive_build_tree(node):
    """
    :param node: a node, with a df within it.
    :return: None.
    """
    all_same_target = len(node.df.loc[node.df.busy == 1]) in [len(node.df), 0]   # check if all the samples are from the same class.
    if len(node.df.columns) > 1 and len(node.df) > 1 and not all_same_target:
        split_by = get_max_entropy_feature(node.df)
        values = list(node.df[split_by].drop_duplicates())
        if len(values) > 1:
            for value in values:
                temp_df = node.df.loc[node.df[split_by] == value]            # split the data for each new branch
                temp_df = temp_df.drop(columns=[split_by])                   # drop the column I splinted by
                new_node = Node(parent=node, df=temp_df, split_value=str(value), split_by=split_by)  # create the new tree
                node.children.append(new_node)
                recursive_build_tree(new_node)
        else:
            node.leaf = True
            node.target = majority_vote(node.df)
    else:
        node.leaf = True
        node.target = majority_vote(node.df)


def chi_child_helper(parent, child):
    """
    :param parent: a tree node
    :param child: a tree node, of course one of the parent children.
    :return: the delta value, in order to sum it with the delta from all the other children.
    """
    pk_hat = len(parent.df.loc[parent.df.busy == 1]) * (len(child.df) / len(parent.df))
    nk_hat = len(parent.df.loc[parent.df.busy == 0]) * (len(child.df) / len(parent.df))
    return ((len(child.df.loc[child.df.busy == 1]) - pk_hat)**2 / pk_hat) + ((len(child.df.loc[child.df.busy == 0]) - nk_hat)**2 / nk_hat)


def prune_node(node):
    """
    check if the split in this node was random.
    :param node: a tree node.
    :return: None.
    """
    if node.children and node.parent is not None:
        chi_stat = 0
        for child in node.children:
            chi_stat += chi_child_helper(parent=node, child=child)
        alpha = 0.05
        degrees_of_freedom = len(node.df) - 1
        critical_value = stats.chi2.ppf(1 - alpha, degrees_of_freedom)
        if not chi_stat > critical_value:
            node.children = []
            node.leaf = True
            node.target = majority_vote(node.df)


def prune_tree(node):
    """
    :param node: a tree node, the root of the tree.
    :return:
    """
    global visited
    global pruned_up
    visited = set()
    pruned_up = set()
    find_leafs_and_prune(node)


def prune_up(node):
    """
    climb up the tree and prune.
    :param node: a node.
    :return: None.
    """
    if node not in pruned_up:
        pruned_up.add(node)
        if node.parent is not None:
            if node.parent not in pruned_up:
                prune_node(node.parent)
                prune_up(node.parent)


def find_leafs_and_prune(node):
    """
    searching the leaf nodes, using DFS.
    :param node: a tree node.
    :return:
    """
    if node not in visited:
        visited.add(node)
    for child in node.children:
        if child not in visited and not child.leaf:
            find_leafs_and_prune(child)
        elif child.leaf:
            prune_up(node)


def predict(row, tree):
    """
    :param row: a row from the test df, in a dict form.
    :param tree: the root of the tree.
    :return: predication of the target 1/0
    """
    current = tree
    while not current.leaf:
        found_value = False
        for child in current.children:
            if str(child.split_value) == str(row[child.split_by]):
                found_value = True
                current = child
                break
        if not found_value:
            current.leaf = True
            current.target = majority_vote(tree.df)
    return current.target


def predict_and_check(tree, df_test):
    """
    :param tree: the root of the tree.
    :param df_test: df.
    :return: accuracy score.
    """
    true_predications = 0
    for index, row in df_test.iterrows():
        if str(predict(row=row, tree=tree)) == str(row.busy):
            true_predications += 1
    return round(true_predications / len(df_test), 4)


def build_tree(ratio):
    """
    :param ratio: ratio of the data to use on the train phase.
    :return: None, printing the accuracy.
    """
    df = get_the_ready_df()
    df_train = df.sample(frac=ratio)
    df_test = df.drop(df_train.index)
    tree_with_pruning = Node(parent=None, df=df_train)
    tree_without_pruning = Node(parent=None, df=df_train)
    recursive_build_tree(tree_with_pruning)
    recursive_build_tree(tree_without_pruning)
    prune_tree(tree_with_pruning)
    accuracy_with_pruning = predict_and_check(tree=tree_with_pruning, df_test=df_test)
    accuracy_without_pruning = predict_and_check(tree=tree_without_pruning, df_test=df_test)
    if accuracy_with_pruning > accuracy_without_pruning:
        tree = tree_with_pruning
        acc = accuracy_with_pruning
    else:
        tree = tree_without_pruning
        acc = accuracy_without_pruning
    pptree.print_tree(current_node=tree, horizontal=True)
    print(acc)



def tree_error(k):
    """
    Implement k-fold validation on the decision tree.
    :param k: number of folds to check
    :return: None, printing the accuracy.
    """
    if k < 2:
        print("k can not be smaller than 2.")
    else:
        df = get_the_ready_df()
        df = df.sample(frac=1)  # shuffle the data set
        acc_pruning = 0
        acc_no_pruning = 0
        for method in ['pruning', 'no_pruning']:
            scores = []
            for i in range(0, len(df), round(len(df)/k)):
                df_train = df.iloc[i:i+round(len(df)/k), :]
                df_test = df.drop(df_train.index)
                tree = Node(parent=None, df=df_train)
                recursive_build_tree(tree)
                if method == 'pruning':
                    prune_tree(tree)
                scores.append(predict_and_check(tree=tree, df_test=df_test))
                if method == 'pruning':
                    acc_pruning = sum(scores) / len(scores)
                else:
                    acc_no_pruning = sum(scores) / len(scores)
        print(acc_pruning) if acc_pruning > acc_no_pruning else print(acc_no_pruning)


def is_busy(row_input):
    """
    :param row_input: an array, in the same form as in the train data set, but without the "Rented Bike Count" column.
    :return: 1 for predication of a busy day, else 0.
    example input I checked:
    row =  ['1/12/2017', 12, 20, 30, 3, 2000, -7, 0, 0, 0, 'Winter', 'No Holiday', 'Yes']
    row = ['21/06/2018', 18, 27.8, 43, 3, 1300, 15, 0.56, 0, 0, 'Summer', 'Holiday', 'Yes']
    """
    row_input = [row_input[0]] + [0] + row_input[1:]      # add a cell to the array, where the "Rented Bike Count" should have been.
    df = pd.read_csv(f"SeoulBikeData.csv", encoding='unicode_escape')
    df_row = pd.DataFrame([], columns=list(df.columns))   # create a df for the row input
    df_row.loc[len(df_row)] = row_input                   # add the row input to the df
    # change the row types to the same types as in the csv file
    for column in ['Rented Bike Count', 'Hour', 'Humidity(%)', 'Visibility (10m)']:
        df_row[column] = df_row[column].astype('int64')
    for column in ['Temperature(°C)', 'Wind speed (m/s)', 'Dew point temperature(°C)', 'Solar Radiation (MJ/m2)', 'Rainfall(mm)', 'Snowfall (cm)']:
        df_row[column] = df_row[column].astype('float64')
    # change the input row in the same way I changed the data set the model trained on.
    df_row = get_the_ready_df(df=df_row)
    tree = Node(parent=None, df=get_the_ready_df())
    recursive_build_tree(tree)
    # recursive_prune_tree(tree)
    for index, row in df_row.iterrows():
        return 1 if predict(row=row, tree=tree) == 1 else 0

