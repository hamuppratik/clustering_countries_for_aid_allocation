from flask import Flask, request, render_template, jsonify
import numpy as np
import pandas as pd
import pickle


app = Flask(__name__)

# Load the Model-(KMeans clustering random state = 42, number of clusters = 3)
model_pickle = open("artifacts/model.pkl", "rb")
model = pickle.load(model_pickle)

# Loading Std scaler for the scaling of Test data
Scaler_pickle = open("artifacts/scaler.pkl", "rb")
scaler = pickle.load(Scaler_pickle)

Templte_path = "index.html"

@app.route('/')
def home():
    return render_template(Templte_path)

# -------------------------------------------------------
@app.route("/predict", methods=["POST"]) 
def predict_manual():
    try:

        data = {
            "country": request.form["country"],
            "child_mort": float(request.form["child_mort"]),
            "exports": float(request.form["exports"]),
            "health": float(request.form["health"]),
            "imports": float(request.form["imports"]),
            "income": float(request.form["income"]),
            "inflation": float(request.form["inflation"]),
            "life_expec": float(request.form["life_expec"]),
            "total_fer": float(request.form["total_fer"]),
            "gdpp": float(request.form["gdpp"]),
        }

        # Get data in DataFrame
        df = pd.DataFrame([data])
        
        # Feature engineering as before
        df["exports_per_capita"] = df["exports"]/100*df["gdpp"]
        df["imports_per_capita"] = df["imports"]/100*df["gdpp"]
        df["health_spending"] = df["health"]/100*df["gdpp"]
        df["High_Child_Mortality"] = (df["child_mort"] > 20).astype(int)
        df["low_Life_Expectancy"] = (df["life_expec"] < 73.1).astype(int)
        df["ratio_export_import"] = df["exports"]/df["imports"]
        df["inflation_adjusted_gdpp"] = df["gdpp"] / (1 + df["inflation"])

        num_cols = ['child_mort', 'exports', 'health', 'imports', 'income',
            'inflation', 'life_expec', 'total_fer', 'gdpp', 'exports_per_capita',
            'imports_per_capita', 'health_spending', 'High_Child_Mortality',
            'low_Life_Expectancy', 'ratio_export_import',
            'inflation_adjusted_gdpp']
        
        X = scaler.transform(df[num_cols])  

        # Prediction
        cluster = model.predict(X)[0]

        # Map cluster with string Top, Poor, Bourgeoisie
        cluster_map = {
            0: "Very Poor Class",
            1: "Bourgeoisie Class",
            2: "Very Rich or one of the Top Class",
        }

        # calculating the distance from cluster centroids and find important features for class
        cluster_centroids = model.cluster_centers_
        distances = [np.linalg.norm(X - centroid) for centroid in cluster_centroids]
        closest_centroid_index = np.argmin(distances)
        closest_centroid = cluster_centroids[closest_centroid_index]

        # Calculate the feature importance (you can use feature distance to centroid for simplicity)
        feature_importance = {}
        for i, feature in enumerate(num_cols):
            feature_importance[feature] = abs(X[0][i] - closest_centroid[i])

        # Sort features by importance
        sorted_features = sorted(feature_importance.items(), key=lambda x: x[1], reverse=True)

        # Select top 3 features (or more depending on your requirement)
        important_features = sorted_features[:3]

        # Prepare the result with important features
        result = {
            "Country": data["country"],
            "Cluster": cluster_map.get(cluster, "Unknown Class"),
            "Key Factors": {feature: value for feature, value in important_features}
        }

        return render_template("index.html", message="Prediction successful", results=[result])

    except Exception as e:
        return render_template("index.html", error=f"An error occurred: {str(e)}")

# -------------------------------------------------------
    
@app.route('/upload', methods=['POST'])
def upload():
    try:
        # Step 1: Get the data using request
        uploaded_file = request.files['data']

        # Step 2: Read into DataFrame
        df = pd.read_csv(uploaded_file)

        # Feature Engineering    
        df["exports_per_capita"] = df["exports"]/100*df["gdpp"]
        df["imports_per_capita"] = df["imports"]/100*df["gdpp"]
        df["health_spending"] = df["health"]/100*df["gdpp"]
        df["High_Child_Mortality"] = (df["child_mort"] > 20).astype(int)
        df["low_Life_Expectancy"] = (df["life_expec"] < 73.1).astype(int)
        df["ratio_export_import"] = df["exports"]/df["imports"]
        df["inflation_adjusted_gdpp"] = df["gdpp"] / (1 + df["inflation"])

        num_cols = ['child_mort', 'exports', 'health', 'imports', 'income',
                    'inflation', 'life_expec', 'total_fer', 'gdpp', 'exports_per_capita',
                    'imports_per_capita', 'health_spending', 'High_Child_Mortality',
                    'low_Life_Expectancy', 'ratio_export_import',
                    'inflation_adjusted_gdpp']

        # Transform features for prediction
        Y_test = scaler.transform(df[num_cols])
        
        # Predict clusters
        clusters = model.predict(Y_test)
        
        # Get countries list
        countries = df["country"].to_list()

        # Calculate the most influential features for each country
        cluster_centroids = model.cluster_centers_
        results = []
        for country, cluster, features in zip(countries, clusters, Y_test):
            distances = [np.linalg.norm(features - centroid) for centroid in cluster_centroids]
            closest_centroid_index = np.argmin(distances)
            closest_centroid = cluster_centroids[closest_centroid_index]
            
            # Calculate feature importance (distance to centroid)
            feature_importance = {}
            for i, feature in enumerate(num_cols):
                feature_importance[feature] = abs(features[i] - closest_centroid[i])

            # Sort features 
            sorted_features = sorted(feature_importance.items(), key=lambda x: x[1], reverse=True)
            
            # top 3 influential features
            important_features = sorted_features[:3]

            # results 
            result = {
                "Country": country,
                "Cluster": (
                    "Bourgeoisie Class" if int(cluster) == 1
                    else "Very Poor Class" if int(cluster) == 0 
                    else "Very Rich or one of the Top Class"
                ),
                "Key Factors": {feature: value for feature, value in important_features}
            }
            results.append(result)

        # Message
        message = "File processed successfully! Check the predicted clusters below."

        # Step 7: Render 
        return render_template(
            'index.html', 
            message=message,
            results=results
        )
    
    except (KeyError, ValueError) as e:
        error_message = """
        Incorrect input format. Please follow the correct format below:
        
        **For CSV Upload**:
        Ensure the file contains the following columns in the exact order: 
        "country", "child_mort", "exports", "health", "imports", "income", "inflation", "life_expec", "total_fer", "gdpp".
        
        **For Manual Input**: 
        "country":"United States",
        "child_mort":7.3,
        "exports":12.4,
        "health":11.6,
        "imports":17.8,
        "income":49400,
        "inflation":1.22,
        "life_expec":78.7,
        "total_fer":1.93,
        "gdpp":48400
        """

        return render_template("index.html", error=error_message)

    except Exception as e:
        return render_template("index.html", error=f"An unexpected error occurred: {str(e)}")


        
        
if __name__ == '__main__':
    app.run(debug=True)


