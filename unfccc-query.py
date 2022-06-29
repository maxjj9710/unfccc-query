# Importing packages

import unfccc_di_api
import pandas as pd
import treelib as tl
import numpy as np


# Initializing API Readers for both Annex I and Non-Annex I inventory data


annex1_reader = unfccc_di_api.UNFCCCSingleCategoryApiReader(party_category='annexOne')
nannex1_reader = unfccc_di_api.UNFCCCSingleCategoryApiReader(party_category='nonAnnexOne')
print('API Connected')


##########################################
# Units
def get_unit_df():
    """
    Fetches all units present in both annex readers.
    :return: Pandas dataframe containing all units from the API.
    """
    print("Beginning to fetch units from API")
    result = pd.concat([annex1_reader.units, nannex1_reader.units], ignore_index=False).drop_duplicates() # combining results from both inventory
    print("Finished fetching units from API")
    return result

##########################################
# Gases
def get_gas_df():
    """
    Fetches all gases present in both annex readers.
    :return: Pandas dataframe containing all gases from the API.
    """
    print("Beginning to fetch gases from API")
    result = pd.concat([annex1_reader.gases, nannex1_reader.gases], ignore_index=False).drop_duplicates() # combining results from both inventory
    print("Finished fetching gases from API")
    return result


##########################################
# Classifications
def get_class_df():
    """
    Fetches all classifications present in both annex readers.
    :return: Pandas dataframe containing all classifications from the API.
    """
    print("Beginning to fetch classifications from API")
    result = pd.concat([annex1_reader.classifications, nannex1_reader.classifications], ignore_index=False).drop_duplicates() # combining results from both inventory
    print("Finished fetching classifications from API")
    return result


##########################################
# Years
def get_year_df():
    """
    Fetches all years present in both annex readers.
    :return: Pandas dataframe containing all years from the API.
    """
    print("Beginning to fetch years from API")
    result = pd.concat([annex1_reader.years, nannex1_reader.years],
                       ignore_index=False).drop_duplicates()  # combining results from both inventory
    print("Finished fetching years from API")
    result.rename(columns={'name': '[YEAR]'}, inplace=True)

    last_inv_yr = result.iloc[len(result) - 1, result.columns.get_loc("[YEAR]")]

    if len(last_inv_yr) > 4:
        last_inv_yr = last_inv_yr[last_inv_yr.index('(') + 1: last_inv_yr.index(')')]
        result.iloc[len(result) - 1, result.columns.get_loc("[YEAR]")] = last_inv_yr

    return result


##########################################
# Annex
# Not initially part of the API, but could be useful for differentiation for databases
def get_annex_df():
    """
    Fetches annex and non-annex list from the API.
    :return: Pandas dataframe containing annex and non-annex list from the API.
    """
    print("Beginning to fetch annex from API")
    annex_array = np.array([[0, 'Non-Annex I', 'Countries considered by UN as developing countries'],
                            [1, 'Annex I', 'Countries considered by UN as developed countries']])
    result = pd.DataFrame(annex_array, columns=['AnnexID', 'AnnexName', 'AnnexDescr'])
    print("Finished fetching annex from API")
    return result



##########################################
# Parties (Countries)
def get_party_df():
    """
    Fetches countries from annex and non-annex list from the API.
    :return: Pandas dataframe containing countries from annex and non-annex list in the API.
    """
    print("Beginning to fetch parties from API")
    n_parties = nannex1_reader.parties
    a_parties = annex1_reader.parties

    # appending foreign key
    n_parties['AnnexID'] = 0
    a_parties['AnnexID'] = 1
    result = pd.concat([n_parties, a_parties], ignore_index=False).drop_duplicates().drop('noData', axis=1)
    print("Finished fetching parties from API")
    return result



##########################################
# Measurement Types
# API presents measurements as treelib object, and each measurement contains a supertype. Because there is no default
# root for the measurement treelib, the auto-generated root needs to be referenced every time the API reader is called.
#
# NOTE: Auto-increment is not used because each measurement has a specific ID used for querying, which has been set as
#       the primary key

# Annex I
def get_measure_type_df():
    print("Beginning to fetch measurement types from API")
    a_mt_root = annex1_reader.measure_tree.all_nodes()[0].identifier  # referencing root
    a_mt_nodes = annex1_reader.measure_tree.children(a_mt_root)
    a_mt_array = []

    for node in a_mt_nodes:
        a_mt_array.append([node.identifier, node.tag])

    a_mt_df = pd.DataFrame(np.array(a_mt_array), columns=['MeasureTypeID', 'MeasureTypeName'])

    # Non-Annex I
    n_mt_root = nannex1_reader.measure_tree.all_nodes()[0].identifier  # referencing root
    n_mt_nodes = nannex1_reader.measure_tree.children(n_mt_root)
    n_mt_array = []
    for node in n_mt_nodes:
        n_mt_array.append([node.identifier, node.tag])

    n_mt_df = pd.DataFrame(np.array(n_mt_array), columns=['MeasureTypeID', 'MeasureTypeName'])

    # Combined
    result = pd.concat([a_mt_df, n_mt_df], ignore_index=False).drop_duplicates().set_index('MeasureTypeID')
    print("Finished fetching measurement types from API")
    return result


##########################################
# Measurements
# Using previously found measurement supertypes to look for children (measurements)
#
# NOTE: Auto-increment is not used because each measurement has a specific ID used for querying, which has been set as
#       the primary key

# Annex I
def get_measure_df(all_mt):
    print("Beginning to fetch measurements from API")
    a_measure_array = []

    for mt in all_mt.index.tolist():
        children = annex1_reader.measure_tree.children(int(mt))

        for node in children:
            a_measure_array.append([node.identifier, int(mt), node.tag])

    a_measure_df = pd.DataFrame(np.array(a_measure_array), columns=['MeasureID', 'MeasureTypeID', 'MeasureName'])
    a_measure_df['AnnexID'] = 1

    # Non-Annex I
    n_measure_array = []

    for mt in all_mt.index.tolist():
        children = nannex1_reader.measure_tree.children(int(mt))

        for node in children:
            n_measure_array.append([node.identifier, int(mt), node.tag])

    n_measure_df = pd.DataFrame(np.array(n_measure_array), columns=['MeasureID', 'MeasureTypeID', 'MeasureName'])
    n_measure_df['AnnexID'] = 0

    # Combined
    result = pd.concat([a_measure_df, n_measure_df], ignore_index=False).drop_duplicates().set_index('MeasureID')
    print("Finished fetching measurements from API")
    return result



##########################################
# Categories
# API presents categories as treelib object, hence the whole tree needs to be traversed to obtain all categories.
#
# NOTE: Same as measurements, each category has a specific ID used for querying, which has been set as the primary key.


def format_category(cid, tree, curr_array, depth):
    """
    Recursively traverses the category treelib object based on given reader. If there exists children for the current
    node, append the data to curr_array and call its children; otherwise move on to the next node.
    :param cid: Query ID for the current node
    :param tree: Deep copied category treelib object
    :param curr_array: Current array containing all categories
    :param depth: Depth level of current node with respect to the treelib
    :return: Array contain all categories in the passed-in API reader
    """
    depth = depth + 1
    nodes = tree.children(cid)

    if nodes:
        for node in nodes:
            tag_split = node.tag.split('  ')
            if len(tag_split) == 1:
                tag_split.insert(0, '')
                if depth > 0:
                    parent_tag = tree.get_node(cid).tag.split('  ')[0]
                    if parent_tag[-1] == '.' :
                        tag_split[0] = parent_tag + tag_split[1]
                    else:
                        tag_split[0] = parent_tag + '.' + tag_split[1]
                    tree.update_node(node.identifier, tag=(tag_split[0] + '  ' + tag_split[1]))
                    print(tag_split)
            curr_array = np.vstack([curr_array, np.array([int(node.identifier), tag_split[0], tag_split[1], depth])])
            curr_array = format_category(node.identifier, tree, curr_array, depth)

    return curr_array

def get_category_df():
    print("Beginning to fetch categories from API")
    a_root_array = np.array([annex1_reader.category_tree.all_nodes()[0].identifier, '', 'Totals', 0])
    n_root_array = np.array([nannex1_reader.category_tree.all_nodes()[0].identifier, '', 'Totals', 0])

    a_deep_tree = tl.tree.Tree(tree=annex1_reader.category_tree, deep=True)
    n_deep_tree = tl.tree.Tree(tree=nannex1_reader.category_tree, deep=True)

    a_formatted_category = format_category(int(a_root_array[0]), a_deep_tree, a_root_array, 0)
    n_formatted_category = format_category(int(n_root_array[0]), n_deep_tree, n_root_array, 0)

    a_category_df = pd.DataFrame(a_formatted_category, columns=['CategoryID', 'Tag', 'CategoryName', 'Depth'])
    a_category_df['AnnexID'] = 1
    n_category_df = pd.DataFrame(n_formatted_category, columns=['CategoryID', 'Tag', 'CategoryName', 'Depth'])
    n_category_df['AnnexID'] = 0

    result_df, a_category_length = pd.concat([a_category_df, n_category_df], ignore_index=False).\
                                       drop_duplicates().set_index('CategoryID'), len(a_category_df)
    print("Finished fetching categories from API")
    return result_df, a_category_length


##########################################
# Queries
# Using party codes from both readers and Category IDs to query all data values from the API.
# Runtime of the queries could be >30 minutes.

def get_query_df(all_category, annex_cutoff):
    print("Beginning to fetch queries from API")
    a_party_codes = annex1_reader.parties['code']
    n_party_codes = nannex1_reader.parties['code']
    category_ids = list(map(int, all_category['CategoryID']))

    query_df = pd.DataFrame()

    for cid in category_ids[:annex_cutoff]:
        try:
            query = annex1_reader.query(party_codes=a_party_codes, category_ids=[cid])
        except unfccc_di_api.NoDataError:
            continue
        else:
            query_df = pd.concat([query_df, query], ignore_index=True)

    for cid in category_ids[annex_cutoff:]:
        try:
            query = nannex1_reader.query(party_codes=n_party_codes, category_ids=[cid])
        except unfccc_di_api.NoDataError:
            continue
        else:
            query_df = pd.concat([query_df, query], ignore_index=True)

    print("Finished fetching queries from API")
    return query_df


def main():
    # unit_df = get_unit_df()
    gas_df = get_gas_df()
    # class_df = get_class_df()
    # year_df = get_year_df()
    # annex_df = get_annex_df()
    # party_df = get_party_df()
    # measure_type_df = get_measure_type_df()
    # measure_type_df.to_csv('./all_measures_types.csv')
    # measure_df = get_measure_df(measure_type_df)
    category_df, annex_cutoff = get_category_df()
    print(category_df)
    # query_df = get_query_df(category_df, annex_cutoff)
    #
    # unit_df.to_csv('./units.csv', header=True)
    # gas_df.to_csv('./gases.csv', header=True)
    # class_df.to_csv('./classifications.csv')
    # year_df.to_csv('./years.csv')
    # annex_df.to_csv('./annexes.csv')
    # party_df.to_csv('./parties.csv')
    #
    # measure_df.to_csv('./measures.csv')
    # category_df.to_csv('./categories.csv')
    # query_df.to_csv('./queries.csv')


main()
