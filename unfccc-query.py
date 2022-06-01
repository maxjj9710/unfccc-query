# Importing packages

import unfccc_di_api
import pandas as pd
import numpy as np


# Initializing API Readers for both Annex I and Non-Annex I inventory data

annex1_reader = unfccc_di_api.UNFCCCSingleCategoryApiReader(party_category='annexOne')
nannex1_reader = unfccc_di_api.UNFCCCSingleCategoryApiReader(party_category='nonAnnexOne')
print('API Connected')


##########################################
# Units

all_units = pd.concat([annex1_reader.units, nannex1_reader.units], ignore_index=False).drop_duplicates() # combining results from both inventory
all_units.to_csv('./units.csv', header=True)


##########################################
# Gases

all_gases = pd.concat([annex1_reader.gases, nannex1_reader.gases], ignore_index=False).drop_duplicates() # combining results from both inventory
all_gases.to_csv('./gases.csv', header=True)


##########################################
# Classifications

all_classifications = pd.concat([annex1_reader.classifications, nannex1_reader.classifications], ignore_index=False).drop_duplicates() # combining results from both inventory
all_classifications.to_csv('./classifications.csv')


##########################################
# Years

all_years = pd.concat([annex1_reader.years, nannex1_reader.years], ignore_index=False).drop_duplicates() # combining results from both inventory
all_years.to_csv('./years.csv')


##########################################
# Annex
# Not initially part of the API, but could be useful for differentiation for databases

annex_array = np.array([[0, 'Non-Annex I', 'Countries considered by UN as developing countries'],
                        [1, 'Annex I', 'Countries considered by UN as developed countries']])
annex_df = pd.DataFrame(annex_array, columns=['AnnexID', 'AnnexName', 'AnnexDescr'])
annex_df.set_index('AnnexID').to_csv('./annexes.csv')


##########################################
# Parties (Countries)

n_parties = nannex1_reader.parties
a_parties = annex1_reader.parties

# appending foreign key
n_parties['AnnexID'] = 0
a_parties['AnnexID'] = 1
all_parties = pd.concat([n_parties, a_parties], ignore_index=False).drop_duplicates().drop('noData', axis=1)
all_parties.to_csv('./parties.csv')


##########################################
# Measurement Types
# API presents measurements as treelib object, and each measurement contains a supertype. Because there is no default
# root for the measurement treelib, the auto-generated root needs to be referenced every time the API reader is called.
#
# NOTE: Auto-increment is not used because each measurement has a specific ID used for querying, which has been set as
#       the primary key

# Annex I
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
all_mt = pd.concat([a_mt_df, n_mt_df], ignore_index=False).drop_duplicates()
all_mt.set_index('MeasureTypeID').to_csv('./all_measures_types.csv')


##########################################
# Measurements
# Using previously found measurement supertypes to look for children (measurements)
#
# NOTE: Auto-increment is not used because each measurement has a specific ID used for querying, which has been set as
#       the primary key

# Annex I
a_measure_array = []

for mt in all_mt['MeasureTypeID']:
    children = annex1_reader.measure_tree.children(int(mt))

    for node in children:
        a_measure_array.append([node.identifier, int(mt), node.tag])

a_measure_df = pd.DataFrame(np.array(a_measure_array), columns=['MeasureID', 'MeasureTypeID', 'MeasureName'])

# Non-Annex I
n_measure_array = []

for mt in all_mt['MeasureTypeID']:
    children = nannex1_reader.measure_tree.children(int(mt))

    for node in children:
        n_measure_array.append([node.identifier, int(mt), node.tag])

n_measure_df = pd.DataFrame(np.array(n_measure_array), columns=['MeasureID', 'MeasureTypeID', 'MeasureName'])

# Combined
all_measures = pd.concat([a_measure_df, n_measure_df], ignore_index=False).drop_duplicates()
all_measures.set_index('MeasureID').to_csv('./measures.csv')


##########################################
# Categories
# API presents categories as treelib object, hence the whole tree needs to be traversed to obtain all categories.
#
# NOTE: Same as measurements, each category has a specific ID used for querying, which has been set as the primary key.


def format_category(cid, reader, curr_array, depth):
    """
    Recursively traverses the category treelib object based on given reader. If there exists children for the current
    node, append the data to curr_array and call its children; otherwise move on to the next node.
    :param cid: Query ID for the current node
    :param reader: API reader
    :param curr_array: Current array containing all categories
    :param depth: Depth level of current node with respect to the treelib
    :return: array contain all categories in the passed-in API reader
    """
    depth = depth + 1
    nodes = reader.category_tree.children(cid)

    if nodes:
        for node in nodes:
            tag_split = node.tag.split('  ')
            if len(tag_split) == 1:
                tag_split.insert(0, '')
            curr_array = np.vstack([curr_array, np.array([int(node.identifier), tag_split[0], tag_split[1], depth])])
            curr_array = format_category(node.identifier, reader, curr_array, depth)

    return curr_array


a_root_array = np.array([annex1_reader.category_tree.all_nodes()[0].identifier, '', 'Totals', 0])
n_root_array = np.array([nannex1_reader.category_tree.all_nodes()[0].identifier, '', 'Totals', 0])
a_formatted_category = format_category(int(a_root_array[0]), annex1_reader, a_root_array, 0)
n_formatted_category = format_category(int(n_root_array[0]), nannex1_reader, n_root_array, 0)

a_category_df = pd.DataFrame(a_formatted_category, columns=['CategoryID', 'Tag', 'CategoryName', 'Depth'])
n_category_df = pd.DataFrame(n_formatted_category, columns=['CategoryID', 'Tag', 'CategoryName', 'Depth'])

all_category = pd.concat([a_category_df, n_category_df], ignore_index=False).drop_duplicates()
all_category.set_index('CategoryID').to_csv('./categories.csv')


##########################################
# Queries
# Using party codes from both readers and Category IDs to query all data values from the API.
# Runtime of the queries could be >30 minutes.

a_party_codes = annex1_reader.parties['code']
n_party_codes = nannex1_reader.parties['code']
category_ids = list(map(int, all_category['CategoryID']))

query_df = pd.DataFrame()

for cid in category_ids[:len(a_category_df)]:
    try:
        query = annex1_reader.query(party_codes=a_party_codes, category_ids=[cid])
    except unfccc_di_api.NoDataError:
        continue
    else:
        query_df = pd.concat([query_df, query], ignore_index=True)

for cid in category_ids[len(a_category_df):]:
    try:
        query = nannex1_reader.query(party_codes=n_party_codes, category_ids=[cid])
    except unfccc_di_api.NoDataError:
        continue
    else:
        query_df = pd.concat([query_df, query], ignore_index=True)

query_df.to_csv('queries_test.csv')



