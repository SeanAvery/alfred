from keras.models import Sequential
from keras.layers import Dense
from keras.optimizers import Adam

class CartPoleAgent():
    def build_model(self):
        model = Sequential()
        model.add(Dense(12, activation='relu'))
        model.add(Dense(12, activation='relu'))
        model.add(Dense(2))
        model.compile(Adam(lr=0.001), 'mse')

        return model


if __name__ == '__main__':
    agent = CartPoleAgent()
    agent.build_model()
