# zomato-data-analysis
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
import seaborn as sns

class ZomatoAnalyzer:
    """
    A class to analyze Zomato restaurant data
    """
    
    def __init__(self, df):
        """Initialize with a pandas DataFrame"""
        self.df = df
        self.results = {}
        
    def analyze_basic_metrics(self):
        """Calculate basic statistical metrics"""
        self.results['basic_metrics'] = {
            'total_restaurants': len(self.df),
            'average_rating': self.df['rating'].mean(),
            'median_cost': self.df['cost_for_two'].median(),
            'cuisine_diversity': self.df['cuisines'].nunique(),
            'locations_covered': self.df['location'].nunique()
        }
        
    def analyze_pricing(self):
        """Analyze restaurant pricing patterns"""
        # Create price brackets
        self.df['price_category'] = pd.qcut(
            self.df['cost_for_two'],
            q=4,
            labels=['Budget', 'Medium', 'High', 'Premium']
        )
        
        self.results['pricing_analysis'] = {
            'price_distribution': self.df['price_category'].value_counts().to_dict(),
            'avg_rating_by_price': self.df.groupby('price_category')['rating'].mean().to_dict(),
            'price_location_correlation': self.df.groupby('location')['cost_for_two'].mean().sort_values(ascending=False).head(10).to_dict()
        }
        
    def analyze_cuisines(self):
        """Analyze cuisine patterns and preferences"""
        # Split multiple cuisines and create a flat list
        all_cuisines = self.df['cuisines'].str.split(',').explode().str.strip()
        
        self.results['cuisine_analysis'] = {
            'top_cuisines': all_cuisines.value_counts().head(10).to_dict(),
            'cuisine_ratings': self.df.groupby('cuisines')['rating'].mean().sort_values(ascending=False).head(10).to_dict(),
            'cuisine_avg_cost': self.df.groupby('cuisines')['cost_for_two'].mean().sort_values(ascending=False).head(10).to_dict()
        }
        
    def analyze_locations(self):
        """Analyze geographical patterns"""
        self.results['location_analysis'] = {
            'restaurant_density': self.df['location'].value_counts().head(10).to_dict(),
            'location_ratings': self.df.groupby('location')['rating'].mean().sort_values(ascending=False).head(10).to_dict(),
            'location_cost_profile': self.df.groupby('location')['cost_for_two'].agg(['mean', 'median', 'std']).to_dict()
        }
        
    def perform_clustering(self):
        """Perform K-means clustering based on ratings and cost"""
        # Prepare data for clustering
        features = self.df[['rating', 'cost_for_two']].copy()
        scaler = StandardScaler()
        features_scaled = scaler.fit_transform(features)
        
        # Perform k-means clustering
        kmeans = KMeans(n_clusters=4, random_state=42)
        self.df['cluster'] = kmeans.fit_predict(features_scaled)
        
        # Analyze clusters
        self.results['clustering'] = {
            'cluster_sizes': self.df['cluster'].value_counts().to_dict(),
            'cluster_profiles': self.df.groupby('cluster').agg({
                'rating': 'mean',
                'cost_for_two': 'mean',
                'price_category': lambda x: x.mode()[0]
            }).to_dict()
        }
        
    def generate_visualizations(self, save_path='results/figures/'):
        """Generate and save visualization plots"""
        # Set style
        plt.style.use('seaborn')
        
        # 1. Rating Distribution
        plt.figure(figsize=(10, 6))
        sns.histplot(data=self.df, x='rating', bins=20)
        plt.title('Distribution of Restaurant Ratings')
        plt.savefig(f'{save_path}rating_distribution.png')
        plt.close()
        
        # 2. Price vs Rating
        plt.figure(figsize=(10, 6))
        sns.scatterplot(data=self.df, x='cost_for_two', y='rating', alpha=0.5)
        plt.title('Price vs Rating Correlation')
        plt.savefig(f'{save_path}price_vs_rating.png')
        plt.close()
        
        # 3. Top Cuisines
        plt.figure(figsize=(12, 6))
        cuisine_counts = self.df['cuisines'].value_counts().head(10)
        sns.barplot(x=cuisine_counts.values, y=cuisine_counts.index)
        plt.title('Top 10 Cuisine Types')
        plt.savefig(f'{save_path}top_cuisines.png')
        plt.close()
        
    def run_complete_analysis(self):
        """Run all analysis methods and return results"""
        self.analyze_basic_metrics()
        self.analyze_pricing()
        self.analyze_cuisines()
        self.analyze_locations()
        self.perform_clustering()
        self.generate_visualizations()
        
        return self.results

    def generate_report(self):
        """Generate a text report of key findings"""
        if not self.results:
            self.run_complete_analysis()
            
        report = []
        report.append("=== Zomato Restaurant Analysis Report ===\n")
        
        # Basic metrics
        report.append("Basic Metrics:")
        for metric, value in self.results['basic_metrics'].items():
            report.append(f"- {metric.replace('_', ' ').title()}: {value:.2f}")
        
        # Pricing insights
        report.append("\nPricing Insights:")
        report.append(f"- Most common price category: {max(self.results['pricing_analysis']['price_distribution'].items(), key=lambda x: x[1])[0]}")
        report.append(f"- Highest rated price category: {max(self.results['pricing_analysis']['avg_rating_by_price'].items(), key=lambda x: x[1])[0]}")
        
        # Cuisine insights
        report.append("\nCuisine Insights:")
        top_cuisine = max(self.results['cuisine_analysis']['top_cuisines'].items(), key=lambda x: x[1])[0]
        report.append(f"- Most popular cuisine: {top_cuisine}")
        
        # Location insights
        report.append("\nLocation Insights:")
        top_location = max(self.results['location_analysis']['restaurant_density'].items(), key=lambda x: x[1])[0]
        report.append(f"- Most restaurant-dense location: {top_location}")
        
        return "\n".join(report)
