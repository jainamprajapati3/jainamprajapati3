from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
import pickle
import streamlit as st

iris=datasets.load_iris()
# print(iris)
X=iris.data
y=iris.target

x_train,x_test,y_train,y_test=train_test_split(X,y)
lin_reg=LinearRegression()
lin_reg=lin_reg.fit(x_train,y_train)

pickle.dump(lin_reg, open('lin_model.pkl', 'wb'))


lin_model=pickle.load(open('lin_model.pkl', 'rb'))

def classify(num):
    if num<0.5:
        return 'Setosa'
    elif num <1.5:
        return 'Versicolor'
    else:
        return 'Virginica'
def main():
    st.title("Jainam Prajapati")
    html_temp = """
    <div style="background-color:yellow ;padding:10px">
    <h2 style="color:white;text-align:center;font-weight: bold;">Iris Classification</h2>
    </div>
    """
    st.markdown(html_temp, unsafe_allow_html=True)
    activities=['Linear Regression']
    sl=st.slider('Select Sepal Length', 0.0, 10.0)
    sw=st.slider('Select Sepal Width', 0.0, 10.0)
    pl=st.slider('Select Petal Length', 0.0, 10.0)
    pw=st.slider('Select Petal Width', 0.0, 10.0)
    inputs=[[sl,sw,pl,pw]]
    if st.button('Classify'):
            st.success(classify(lin_model.predict(inputs)))

if __name__=='__main__':
    main()
