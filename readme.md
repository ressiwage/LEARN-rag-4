# rag
данный проект я решил выполнить в образовательных целях.

суть проекта -- в создании rag с простым запросом.

## установка
0. установка python 3.11
1. pip install -r requirements.txt
2. установка и запуск ollama (см. https://github.com/ollama/ollama?tab=readme-ov-file#windows-preview)
3. ollama run llama2
4. python main.py

### результат
main.py -- вопрос llama2 "что меня ждет в этой квартире?" касательно квартиры с авито с использованием руберта для эмбеддингов и хромадб для векторного хранилища

data/avito.txt -- файл с описаниемм квартиры на авито

