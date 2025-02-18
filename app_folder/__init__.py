from flask import Flask
import os
from app_folder.config_folder.config import Config_  # Explicit import

def create_app(config_class=Config_):  # Use the class directly
    # Initialize Flask app
    app = Flask(__name__,
                template_folder=os.path.join(os.pardir, 'templates'),
                static_folder=os.path.join(os.pardir, 'static'))
    
    # Load configuration
    app.config.from_object(config_class)

    # Register blueprints
    from app_folder.routes.main_routes import main_bp
    from app_folder.routes.search_routes import search_bp
    app.register_blueprint(main_bp)
    app.register_blueprint(search_bp)

    return app