from demand import predict_customer

# Define the environment
class PricingEnvironment:
    def __init__(self, dataset, daily_demand, initial_price):
        self.products_data = dataset
        self.daily_demand = daily_demand
        print(f"daily demand: {daily_demand}")
        self.initial_price = initial_price

        self.total_days = 30
        self.current_day = 0
        self.current_product_index = 0
        self.total_products = 25
        # print(f"df: {self.products_data}")

    def get_product_state(self, product_id, day):
        # print(f'THIS FROM GET PRODUCT product id {product_id} day {day}')
        state = self.products_data.loc[(self.products_data['product_id'] == product_id) & (self.products_data['day'] == day)]
        state = state.iloc[:, 2:].values  # Convert DataFrame subset to a NumPy array, excluding the first two columns
        # print(f'state from get {state}')
        return state

    def reset(self):
        aggregated_state = self.aggregate_data_for_day(1)
        return aggregated_state
    
    def step(self, action, current_day):
        

        # print(f"Day: {self.current_day} Product: {self.current_product_index} ")
        if current_day == self.total_days:
            done = True
        else:
            done = False

        if not done and current_day < self.total_days:
            next_aggregated_state = self.aggregate_data_for_day(current_day + 1)
        else:
            next_aggregated_state = None

        # Calculate the reward based on the chosen action
        reward, d_revenue, s_revenue, demand = self.calculate_reward(action, current_day)

        

       

        return next_aggregated_state, reward, done, d_revenue, s_revenue, demand

    def calculate_reward(self, action, current_day):

        # Get the predicted demand for the current day
        predicted_demand = self.daily_demand.loc[self.daily_demand['day'] == current_day, 'predicted_demand'].values[0]
        
        # Calculate revenues
        dynamic_revenue = action * predicted_demand
        static_revenue = self.initial_price * predicted_demand

        # Calculate reward as the difference between static and dynamic revenues
        reward =  dynamic_revenue - static_revenue
        # print(f" {current_day} reward {reward} price: {action} ")
        # print(f"demand: {predicted_demand}")
        # print(f"dp: {action} bp: {self.initial_price}")
        return reward, dynamic_revenue, static_revenue, predicted_demand
    
    def aggregate_data_for_day(self, day):
        aggregated_state = []

        for product_id in range(1, self.total_products + 1):
            state = self.get_product_state(product_id, day)
            aggregated_state.extend(state)  # Adjust this based on your aggregation logic

        return aggregated_state