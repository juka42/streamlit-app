import streamlit as st
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import statsmodels.formula.api as smf
import statsmodels.api as sm
from patsy import dmatrices
import numpy as np
import pickle
from scipy.stats import norm
from scipy.stats import t
import locale

locale.setlocale(locale.LC_ALL, 'pt_BR.UTF-8')

#### Carregando o dataframe e o metadados
# metadados = pd.read_pickle('./obj/metadados.pkl')
# df = pd.read_pickle('./obj/df.pkl')
# df = df.iloc[:30,:30]
# df = pd.DataFrame({'p1_uwef_1':[1, 1, 1, 1, 1, 1, 1, 1],
    # 't1t2t3':['Cont', 'Cont', 'Home', 'Home', 'RTW1', 'RTW1', 'RTW2', 'RTW2'],
    # 'comp':[1,1,1,1,1,1,1,1],
    # 'lider': [True, False,True, False,True, False,True, False]
    # })

st.sidebar.image("./pngs-04.png", use_column_width=True)

df_csv = st.sidebar.file_uploader('Arquivo de dados', type=['csv','zip'],accept_multiple_files=False,key="fileUploader")
metadados_csv = st.sidebar.file_uploader('Arquivo de metadados', type=['csv'],accept_multiple_files=False,key="fileUploader")

st.title('Análise RTW')

if (metadados_csv is not None):
    metadados = pd.read_csv(metadados_csv)

    ##### Relação sigla - construto
    construtos = metadados.groupby(['Construto', 'sigla construto'])[['nome_variavel']].count()
    construtos = construtos.reset_index()
    construtos.rename(columns={'sigla construto':'sigla'}, inplace=True)

    construtos.set_index('sigla', inplace = True)
    construtos = construtos.Construto


    # st.write(df.loc[:2,'comp']==True)
    constr = st.sidebar.selectbox('Selecione o construto',list(construtos.index))
    # df.columns[df.columns.str.contains(constr)]


    if df_csv is not None:
        if df_csv.name[-3:] =='csv':
            df = pd.read_csv(df_csv)
        else:
            df = pd.read_csv(df_csv, compression='zip')

        coletas = set([x[:2] for x in df.columns if x.find(constr)>0])
        coleta = st.sidebar.selectbox('Selecione a coleta',coletas)
        prefixo = coleta+'_'+constr
        variaveis = [x for x in df.columns if x.find(prefixo)>=0]
        variaveis = [x for x in variaveis if x[-1].isalpha()] + [x for x in variaveis if x[-1].isnumeric()]
        vary = st.sidebar.selectbox('Selecione a variavel',variaveis)

        option = coleta+'_'+constr

        ### Sufixo da variável - média ou questão i
        suf = vary.split('_')[-1]
        if suf=='med':
            suf_ = 'médio'
        else:
            suf_ = 'Q'+suf

        #### Variáveis para auxliar nos títulos e rótulos de eixos
        pesquisa = vary.split('_')[0]
        sg_construto = constr
        nm_construto = construtos.loc[sg_construto]

        ##### Configurações da figura
        plt.rc('figure', figsize=(8, 6))
        fig, axes = plt.subplots(2, 1, sharex=True)
        plt.subplots_adjust(hspace=0.02)

        # Configurações do intervalo de confiança
        alpha_ = 0.05
        phi_ = norm.cdf(1-alpha_/2)
        dodge = .05 # deslocamento do fator 'líder' no gráfico 2

        #### Dados do gráfico 1
        gb1 = df[df['comp']].groupby('t1t2t3')
        tab1 = gb1[[vary]].count()
        tab1.columns = ['N']
        tab1[vary] = gb1[vary].mean()
        tab1['std_err'] = gb1[vary].std() / (tab1['N']**.5)
        tab1['min'] = tab1[vary] - phi_*tab1['std_err']
        tab1['max'] = tab1[vary] + phi_*tab1['std_err']
        tab1['prec'] = t.ppf(1-alpha_/2, tab1['N'])*tab1['std_err']
        tab1['x'] = np.arange(tab1.shape[0])

        #### Definição do gráfico 1
        axes[0].errorbar(tab1['x'], tab1[vary], yerr=tab1['prec'], fmt='.', color='#C875C4')
        axes[0].set_xlabel('')
        # axes[0].set_ylabel(f'{pesquisa} - {nm_construto}')

        ticks = axes[1].set_xticks(tab1['x'])
        labels = axes[1].set_xticklabels(tab1.index, rotation=0)

        axes[0].set_title(f'{nm_construto} {suf_} por grupo e por liderança (sim/não)')
        if len(vary.split('_')[0])>2:
            axes[0].axhline(y=0, color='r', linestyle=':', linewidth=.5)
            txt_pesq = f'diferença {pesquisa.strip()} - {nm_construto.strip()} {suf_.strip()}'
        else:
            txt_pesq = f'{pesquisa.strip()} - {nm_construto.strip()} {suf_.strip()}'

        fig.text(0.03,0.5, txt_pesq, size=10, rotation=90.,
                 ha="left", va="center")

        ##

        #### Dados do gráfico 2
        gb2 = df[df['comp']].groupby(['t1t2t3', 'lider'])
        tab2 = gb2[[vary]].count()
        tab2.columns = ['N']
        tab2[vary] = gb2[vary].mean()
        tab2['std_err'] = gb2[vary].std() / (tab2['N']**.5)
        tab2['min'] = tab2[vary] - 2*tab2['std_err']
        tab2['max'] = tab2[vary] + 2*tab2['std_err']
        tab2['prec'] = t.ppf(1-alpha_/2, tab2['N'])*tab2['std_err']

        lab_x = list(tab2.index.unique(0))

        tab2['x'] = 0
        n1 = tab2.loc[(slice(None),True), 'x'].shape[0]
        tab2.loc[(slice(None),True), 'x'] = np.arange(n1)-dodge

        n2 = tab2.loc[(slice(None),False), 'x'].shape[0]
        tab2.loc[(slice(None),False), 'x'] = np.arange(n2)+dodge

        colors = {True:1, False:2}
        lider = list(tab2.index.get_level_values(1).map(colors))

        #### Gráfico 2

        cores = ['orange', 'red']

        tab2a = tab2.loc[(slice(None),True),:]
        tab2b = tab2.loc[(slice(None),False),:]

        axes[1].errorbar(tab2a['x'], tab2a[vary], yerr=tab2a['prec']
                         , fmt='.', label = 'Líder', color='#7BC8F6')
        axes[1].errorbar(tab2b['x'], tab2b[vary], yerr=tab2b['prec']
                         , fmt='.', label = 'Não líder', color='#FFA500')
        axes[1].legend(loc='lower center', bbox_to_anchor=(0.5, -.3),
                  ncol=2, fancybox=True, shadow=True)
        # axes[1].set_ylabel(f'{pesquisa} - {nm_construto}')
        if len(vary.split('_')[0])>2:
            axes[1].axhline(y=0, color='r', linestyle=':', linewidth=.5)

        #####
        # axes[1].legend(labels=['Não', 'Sim'])
        # axes[1].legend(loc='upper left')

        st.pyplot(fig)
        # st.plotly_chart(fig)

        def formata(serie_, casas_):
        #     serie.apply(lambda x: locale.format_string('%.2f', x))
            return serie_.apply(lambda x: locale.format_string(f'%.{casas_}f', x))

        def formatadf(df_, casas):
            return df_.apply(lambda x: formata(x, casas))

        #### Tabelas com os dados do gráfico
        st.header(f'Tabela de {nm_construto} por grupo')
        st.table(formatadf(tab1[['N', vary, 'std_err']], 2))


        st.header(f'Tabela de {nm_construto} por grupo e liderança')
        st.table(formatadf(tab2[['N', vary, 'std_err']], 2))

        #### Análise de variância com 1 fator
        st.header('Anova - Grupo vs '+ vary)
        y, dm = dmatrices(vary + ' ~ C(t1t2t3, Treatment("Home"))', df[df['comp']], return_type='dataframe')
        dm.columns = ['Intercepto', 'Contingencia vs home',
               'RTW1 vs home',
               'RTW2 vs home']
        anova = sm.OLS(y, dm, data=df[df['comp']]).fit()
        summ = anova.summary()

        #### Análise de variância com 2 fatores
        st.markdown(summ.tables[0].as_html().replace('.',','), unsafe_allow_html=True)
        st.markdown(summ.tables[1].as_html().replace('.',','), unsafe_allow_html=True)

        st.header('Anova - Grupo vs Lider vs '+ vary)

        y, dm = dmatrices(vary + ' ~ C(t1t2t3, Treatment("Home")) + lider'
                            , df[df['comp']]
                            , return_type='dataframe')
        dm.columns = ['Intercepto', 'Contingencia vs home', 'RTW1 vs home',
               'RTW2 vs home', 'Líder (sim vs não)']

        anova = sm.OLS(y, dm).fit()
        summ = anova.summary()

        # st.write(anova.summary2())
        st.markdown(summ.tables[0].as_html().replace('.',','), unsafe_allow_html=True)
        st.markdown(summ.tables[1].as_html().replace('.',','), unsafe_allow_html=True)
