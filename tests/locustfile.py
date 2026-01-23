"""Load testing configuration for the News Outlet Classifier API."""

from locust import HttpUser, task, between


class NewsClassifierUser(HttpUser):
    """Simulates a user interacting with the News Outlet Classifier API."""

    wait_time = between(1, 3)

    @task(1)
    def health_check(self):
        """Health check endpoint."""
        self.client.get("/health")

    @task(1)
    def root_endpoint(self):
        """Root endpoint."""
        self.client.get("/")

    @task(7)
    def predict_with_slug(self):
        """Predict news outlet using slug text."""
        slugs = [
            "breaking news today election",
            "trump announces new policy",
            "economy faces challenges ahead",
            "weather alert winter storm",
            "sports highlights nfl games",
            "politics latest updates",
            "business market report",
        ]
        import random
        self.client.post("/predict", json={"slug": random.choice(slugs)})
