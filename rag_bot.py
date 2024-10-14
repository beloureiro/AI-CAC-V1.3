import streamlit as st  # type: ignore


def rag_bot_component():
    # st.title("AI-Skills Advisor")

    tab1, tab2, tab3, tab4, tab5 = st.tabs([
        "AI Skills Overview",
        "Quick Wins",
        "Inefficiencies Analysis",
        "Workflow Improvement",
        "Patient Feedback"
    ])

    with tab1:
        st.header("AI Skills Overview")
        st.image("assets/agents/ragbot1.png", caption="AI Skills Overview", use_column_width=True)
        # Adicione mais conteúdo relevante para a visão geral das habilidades do AI

    with tab2:
        st.header("Quick Wins for Patient Satisfaction")
        st.image("assets/agents/ragbot2.png", caption="Quick Wins", use_column_width=True)
        # Adicione mais conteúdo sobre melhorias rápidas para satisfação do paciente

    with tab3:
        st.header("Inefficiencies Analysis")
        st.image("assets/agents/ragbot3.png", caption="Inefficiencies Analysis", use_column_width=True)
        # Adicione mais conteúdo sobre análise de ineficiências

    with tab4:
        st.header("Workflow Improvement Advice")
        st.image("assets/agents/ragbot4.png", caption="Workflow Improvement", use_column_width=True)
        # Adicione mais conteúdo sobre melhorias no fluxo de trabalho

    with tab5:
        st.header("Patient Feedback Display")
        st.image("assets/agents/ragbot5.png", caption="Patient Feedback", use_column_width=True)
        # Adicione mais conteúdo para exibir feedback dos pacientes

    # Adicione mais lógica aqui conforme necessário
        # Adicione mais conteúdo para exibir feedback dos pacientes

    # Adicione mais lógica aqui conforme necessário
