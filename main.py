
from pymongo import MongoClient
from website import create_app



app = create_app()

if __name__ == '__main__':
    print("Starting Lost & Found Flask Application...")
    print("Available routes:")
    print("  GET  /                    - Main page")
    print("  GET  /welcome             - Welcome page")
    print("  POST /signup              - User registration")
    print("  POST /login               - User login")
    print("  GET  /logout              - User logout")
    print("  POST /submit-lost-item    - Submit lost item")
    print("  POST /submit-found-item   - Submit found item")
    print("  GET  /lost-items          - View lost items")
    print("  GET  /found-items         - View found items")
    print("  POST /search              - Search items")
    print("  GET  /api/items/<type>    - API endpoint")
    print("  GET  /test-db             - Test database")
    
    app.run(debug=True, host='0.0.0.0', port=5000)

