import pandas as pd 

class Preprocessor:
    def __init__(self, path) -> None:
        self.path = path
    

    def data_loader(self) -> self:
        self.train_data = pd.read_csv(self.path + '/train.csv')
        self.test_data = pd.read_csv(self.path + '/test.csv')
        
        return self


    def preprocess(self):
        train = self.train_data.copy()
        test = self.test_data.copy()

        interest_map = {'low': 0, 'medium': 1, 'high': 2}
        if 'interest_level' in train.columns and train['interest_level'].dtype == object:
            train['interest_level'] = train['interest_level'].map(interest_map)
        if 'interest_level' in test.columns and test['interest_level'].dtype == object:
            test['interest_level'] = test['interest_level'].map(interest_map)

        self.train_data = train
        self.test_data = test

        return self

    def create_features(self):
        train = self.train_data.copy()
        test = self.test_data.copy()

        feature_names = [
            'Elevator',
            'HardwoodFloors',
            'CatsAllowed',
            'DogsAllowed',
            'Doorman',
            'Dishwasher',
            'NoFee',
            'LaundryinBuilding',
            'FitnessCenter',
            'Pre-War',
            'LaundryinUnit',
            'RoofDeck',
            'OutdoorSpace',
            'DiningRoom',
            'HighSpeedInternet',
            'Balcony',
            'SwimmingPool',
            'LaundryInBuilding',
            'NewConstruction',
            'Terrace',
        ]

        def normalize_features(series: pd.Series) -> pd.Series:
            return (
                series.fillna('')
                .astype(str)
                .str.replace('"', '', regex=False)
                .str.replace('[', '', regex=False)
                .str.replace(']', '', regex=False)
                .str.replace("'", '', regex=False)
                .str.replace(' ', '', regex=False)
            )

        for df in (train, test):
            if 'features' not in df.columns:
                continue
            feats = normalize_features(df['features'])
            for name in feature_names:
                df[name] = feats.str.contains(name).astype(int)

        self.train_data = train
        self.test_data = test

        return self