#%%
import pandas as pd
import sys
import os
import itertools
sys.path.append(os.path.dirname(sys.path[0]))
from ast import literal_eval
from datasetss.ie_hyperion_dataset import find_word_bounds, clean_text
from models.bert_segmenter import BertSegmenter
from models.bert_rep import BertRep
from datasetss.hyperion_dataset import LABELS 
from xlsxwriter import Workbook
#%%
class Analyzer():
    def __init__(self) -> None:
        self.bert_seg = BertSegmenter()
        self.bert_rep = BertRep()

    def df_to_excel(self, df, threshold):
        wb = Workbook("Nuovo_file.xlsx", {'nan_inf_to_errors': True})
        ws = wb.add_worksheet("Analisi")

        title_format = wb.add_format({
            'bold': True,
            'font_size': 15,
            'font_color': 'black',
            'bg_color': '#35CE8D',
            'text_wrap': True,
            'center_across': True,
            'border': True
        })
        wrap_format = wb.add_format({
            'text_wrap': True,
            'font_size': 15,
            'align': 'left',
            'align': 'top',
        })
        default_format = wb.add_format({
            'text_wrap': False,
            'font_size': 15,
            'align': 'left',
            'align': 'top',
        })
        highlight_format = wb.add_format({
            'bg_color': '#fdff32',
            'text_wrap': True,
            'border': True,
            'font_size': 15,
            'align': 'left',
            'align': 'top',
        })

        #self.set_titles( ws, df, title_format)
        for i, col in enumerate(df.columns):
            print(f'{i}: {col}')
            if col in ['età', 'Età', 'age', 'Age', 'index']:
                ws.set_column(i, i, 6)

            elif col == 'Repertorio':
                ws.write_column('Z1', LABELS)
                ws.data_validation(1, i, 1048575, i, {'validate': 'list','source': '$Z$1:$K$24',})
                ws.set_column(i, i, 25)
            elif col in ['Genere', 'Ruolo']:
                ws.set_column(i, i, 15)
            elif col in []:
                #ws.set_column(i, i, ). #se si volgliono aggiungere altre colonne che richiedono una dimensione specifica
                pass

            else:
                ws.set_column(i, i, 30) # default case

            ws.write(0, i, col, title_format)


        index_str = 1
        index_alt = 1
        index_rep = 1
        highlight = False

        for i_row, row in df.iterrows():
            for col_name, item in row.items():
                if col_name == 'Stralcio':  # da cambiare con le colonne aggiunte con predizione modello
                    for i_stralci, stralci in enumerate(item):
                        ws.write(index_str, df.columns.get_loc(col_name), stralci, wrap_format)
                        index_str += 1

                elif col_name == 'Alternative':  # da cambiare con le colonne aggiunte con predizione modello
                    for i_dict, dictio in enumerate(item):
                        dict_str = ''
                        for key, rep in dictio.items():
                            dict_str += f"{key}: {rep}%\n"
                            if list(dictio.keys()).index(key) == 0 and rep <= threshold:
                                highlight = True

                        dict_str = dict_str.rstrip("\n")
                        ws.write(index_alt, df.columns.get_loc(col_name), dict_str, wrap_format)
                        if highlight == True:
                            ws.write(index_alt, df.columns.get_loc(col_name), dict_str, highlight_format)

                        index_alt += 1
                        highlight = False
                elif col_name == 'Repertorio':
                    for i_first_rep, first_rep in enumerate(item):
                        ws.write(index_rep, df.columns.get_loc(col_name), first_rep, wrap_format)
                        index_rep += 1
                else:
                    ws.write(index_str, df.columns.get_loc(col_name), item, default_format)
        
        wb.close()
        return wb

    
    def predict(self, path_file):
        threshold = 50
        extension = os.path.splitext(path_file)[1]

        if extension == '.xlsx':
            df = pd.read_excel(path_file, converters={'Stralci': literal_eval, 'Repertori': literal_eval} )
        elif extension == '.csv':
            df = pd.read_csv(path_file, converters={'Stralci': literal_eval, 'Repertori': literal_eval})
        elif isinstance(path_file, pd.DataFrame):
            df= path_file
        else:
            raise Exception('Extension of file not supported')
        
        #carica csv
        if 'Testo' in df.columns:
            df['Testo'] = df['Testo'].map(clean_text)
            #df['Stralci'] = df['Stralci'].map(lambda x: [clean_text(s) for s in x])
            #df['Bounds'] = df.apply(lambda x: find_word_bounds(x['Stralci'], x['Testo']), axis=1).values.tolist()
        else:
            raise Exception('The uploaded file is missing the "Testo" column')

        #predict
        df['Stralcio'] = df['Testo'].map(self.bert_seg.predict).values.tolist()
        #df['Bounds_predetti'] = df.apply(lambda x: find_word_bounds(x['Stralci_predetti'], x['Testo']), axis=1).values.tolist()

        list_column_alt = []
        list_column_rep = []
        for i, str_list in df['Stralcio'].items(): #str_list is a list of stralci related to one text
            list_dict = self.bert_rep.predict_vector(str_list)
            list_dict_sorted = []
            list_first_rep = []
            for dizio in list_dict:
                sorted_dict = sorted(dizio.items(), key=lambda item: item[1], reverse = True)
                dizio = dict(sorted_dict[:5])
                for key, value in dizio.items():
                    dizio[key] = round(value*100)

                list_dict_sorted.append(dizio)
                list_first_rep.append(next(iter(dizio)))
             
            list_column_alt.append(list_dict_sorted)
            list_column_rep.append(list_first_rep)
             
            #eventuale aggiornamento contatore x progress bar
            str_column_len = len(df['Stralcio'])
            print(f'Testo numero {i+1}/{ str_column_len} denominato')

        df['Repertorio'] = list_column_rep
        df['Alternative'] = list_column_alt

        if 'Ads' not in df.columns:
            df['Ads'] = ''
        

        workbook = self.df_to_excel(df, threshold)

        #salvataggio/esporta file 
# %%
