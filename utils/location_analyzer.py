import pandas as pd
import json
import plotly.express as px
import plotly.graph_objects as go

class LocationAnalyzer:
    """
    Class for analyzing pre-extracted location data.
    It normalizes cities, regions, and abbreviations to a country level.
    """
    
    def __init__(self):
        """Initialize the LocationAnalyzer with normalization maps."""
        self.country_synonyms = self._get_country_synonyms()
        self.city_to_country_map = self._get_city_to_country_map()
        self.country_continent_map = self._get_country_continent_map()
        self.valid_countries = set(self.country_continent_map.keys())
        
        # Add synonyms to valid countries list
        for synonym in self.country_synonyms:
            self.valid_countries.add(synonym)
    
    def get_location_counts(self, pre_extracted_data):
        """
        Counts unique articles that mention each location, based on pre-extracted data.
        
        Args:
            pre_extracted_data (list): List of article dictionaries from the JSON file.
                (e.g., [{"pmid": "...", "extracted_locations": ["USA", "London"]}])
                
        Returns:
            dict: Dictionary with normalized countries as keys and counts
                  (number of unique articles) as values.
        """
        country_articles = {} # Tracks unique PMIDs per country
        
        for article in pre_extracted_data:
            pmid = article.get("pmid", article.get("article_title", "unknown"))
            raw_locations = article.get("extracted_locations", [])
            
            if not raw_locations:
                continue
            
            normalized_countries_for_article = set()
            
            for loc in raw_locations:
                # Normalize the location (e.g., "Lahore" -> "Pakistan", "ES" -> "Spain")
                country = self._get_country_for_location(loc, self.city_to_country_map)

                # We only want to count *valid, normalized countries*
                if country in self.valid_countries:
                    # Normalize synonyms one last time (e.g., "USA" -> "United States")
                    if country in self.country_synonyms:
                        country = self.country_synonyms[country]
                        
                    normalized_countries_for_article.add(country)
            
            # Now, for each unique *country* found in this article, add the pmid
            for country in normalized_countries_for_article:
                if country not in country_articles:
                    country_articles[country] = set()
                country_articles[country].add(pmid)
                
        # Convert the sets of PMIDs to counts
        location_counts = {country: len(pmids) for country, pmids in country_articles.items()}
        
        return location_counts
        
    def _get_country_synonyms(self):
        """
        Get standardized mapping of country synonyms
        
        Returns:
            dict: Dictionary mapping country synonyms to standard names
        """
        return {
            "USA": "United States",
            "U.S.": "United States",
            "U.S.A.": "United States",
            "America": "United States",
            "United States of America": "United States",
            "UK": "United Kingdom", 
            "U.K.": "United Kingdom",
            "Britain": "United Kingdom",
            "Great Britain": "United Kingdom",
            "England": "United Kingdom",
            "Scotland": "United Kingdom",
            "Wales": "United Kingdom",
            "Northern Ireland": "United Kingdom",
            "Russia": "Russian Federation",
            "Mainland China": "China",
            "People's Republic of China": "China",
            "Republic of Korea": "South Korea",
            "Democratic People's Republic of Korea": "North Korea",
            "Islamic Republic of Iran": "Iran",
            "UAE": "United Arab Emirates",
            "Saudi": "Saudi Arabia",
            "Palestine": "Palestinian Territories",
            "Viet Nam": "Vietnam",
            # Added from new JSON
            "ES": "Spain",
            "Türkiye": "Turkey"
        }
        
    def _get_us_states(self):
        """
        Get all US state names for normalization
        
        Returns:
            set: Set of US state names
        """
        return {
            "Alabama", "Alaska", "Arizona", "Arkansas", "California", 
            "Colorado", "Connecticut", "Delaware", "Florida", "Georgia", 
            "Hawaii", "Idaho", "Illinois", "Indiana", "Iowa", 
            "Kansas", "Kentucky", "Louisiana", "Maine", "Maryland", 
            "Massachusetts", "Michigan", "Minnesota", "Mississippi", "Missouri", 
            "Montana", "Nebraska", "Nevada", "New Hampshire", "New Jersey", 
            "New Mexico", "New York", "North Carolina", "North Dakota", "Ohio", 
            "Oklahoma", "Oregon", "Pennsylvania", "Rhode Island", "South Carolina", 
            "South Dakota", "Tennessee", "Texas", "Utah", "Vermont", 
            "Virginia", "Washington", "West Virginia", "Wisconsin", "Wyoming",
            "District of Columbia", "Washington D.C.", "D.C."
        }
        
    def _get_country_for_location(self, location, city_map):
        """
        Map a location to its country
        
        Args:
            location (str): Location name
            city_map (dict): Dictionary mapping cities to countries
            
        Returns:
            str: Country name or the original location if it's a country
        """
        # Check if location is in our city-to-country mapping
        if location in city_map:
            return city_map[location]
            
        # Check if the location is a country (or country synonym)
        synonyms = self._get_country_synonyms()
        if location in synonyms:
            return synonyms[location]
            
        # Use continent mapping to check if it's a country
        continent = self.map_location_to_continent(location)
        if continent != "Unknown":
            return location
        
        # Check other possible mappings
        for country, continent in self._get_country_continent_map().items():
            if country.lower() in location.lower() or location.lower() in country.lower():
                return country
                
        # If we can't find a mapping, return the original location
        return location
        
    def _get_city_to_country_map(self):
        """
        Get mapping of cities and regions to countries.
        Includes base list + new additions from the sample JSON.
        
        Returns:
            dict: Dictionary with cities/regions as keys and countries as values
        """
        city_map = {
            # North America
            "New York": "United States",
            "Chicago": "United States",
            "Los Angeles": "United States",
            "San Francisco": "United States",
            "Washington": "United States",
            "Boston": "United States",
            "Seattle": "United States",
            "Texas": "United States",
            "California": "United States",
            "Florida": "United States",
            "Michigan": "United States",
            "Ohio": "United States",
            "Massachusetts": "United States",
            "Virginia": "United States",
            "Georgia": "United States",
            "Pennsylvania": "United States",
            "New Jersey": "United States",
            "Colorado": "United States",
            "Oregon": "United States",
            "Arizona": "United States",
            "Connecticut": "United States",
            "North Carolina": "United States",
            "South Carolina": "United States",
            "Tennessee": "United States",
            "Wisconsin": "United States",
            "Indiana": "United States",
            "Missouri": "United States",
            "Maryland": "United States",
            "Minnesota": "United States",
            "Louisiana": "United States",
            "Alabama": "United States",
            "Kentucky": "United States",
            "Utah": "United States",
            "Nevada": "United States",
            "Oklahoma": "United States",
            "Iowa": "United States",
            "Toronto": "Canada",
            "Vancouver": "Canada",
            "Montreal": "Canada",
            "Mexico City": "Mexico",
            
            # Europe
            "London": "United Kingdom",
            "Manchester": "United Kingdom",
            "Greater Manchester": "United Kingdom",
            "Liverpool": "United Kingdom",
            "Paris": "France",
            "Marseille": "France",
            "Lyon": "France",
            "Berlin": "Germany",
            "Hamburg": "Germany",
            "Munich": "Germany",
            "Frankfurt": "Germany",
            "Rome": "Italy",
            "Milan": "Italy",
            "Naples": "Italy",
            "Venice": "Italy",
            "Madrid": "Spain",
            "Barcelona": "Spain",
            "Valencia": "Spain",
            "Amsterdam": "Netherlands",
            "Rotterdam": "Netherlands",
            "Brussels": "Belgium",
            "Zurich": "Switzerland",
            "Geneva": "Switzerland",
            "Vienna": "Austria",
            "Moscow": "Russia",
            "St. Petersburg": "Russia",
            "Stockholm": "Sweden",
            "Oslo": "Norway",
            "Copenhagen": "Denmark",
            "Helsinki": "Finland",
            "Athens": "Greece",
            "Prague": "Czech Republic",
            "Budapest": "Hungary",
            "Warsaw": "Poland",
            "Lisbon": "Portugal",
            "Dublin": "Ireland",
            
            # Asia
            "Beijing": "China",
            "Shanghai": "China",
            "Guangzhou": "China",
            "Shenzhen": "China",
            "Hong Kong": "China",
            "Tokyo": "Japan",
            "Osaka": "Japan",
            "Kyoto": "Japan",
            "Seoul": "South Korea",
            "Delhi": "India",
            "Mumbai": "India",
            "Bangalore": "India",
            "Bengaluru": "India",
            "Chennai": "India",
            "Kolkata": "India",
            "Hyderabad": "India",
            "Ahmedabad": "India",
            "Pune": "India",
            "Jaipur": "India",
            "Lucknow": "India",
            "Karachi": "Pakistan",
            "Lahore": "Pakistan",
            "Islamabad": "Pakistan",
            "Bangkok": "Thailand",
            "Singapore": "Singapore",
            "Kuala Lumpur": "Malaysia",
            "Jakarta": "Indonesia",
            "Tehran": "Iran",
            "Baghdad": "Iraq",
            "Tel Aviv": "Israel",
            "Jerusalem": "Israel",
            "Istanbul": "Turkey",
            "Ankara": "Turkey",
            "Dubai": "United Arab Emirates",
            "Abu Dhabi": "United Arab Emirates",
            
            # Oceania
            "Sydney": "Australia",
            "Melbourne": "Australia",
            "Brisbane": "Australia",
            "Perth": "Australia",
            "Auckland": "New Zealand",
            "Wellington": "New Zealand",
            
            # Africa
            "Cairo": "Egypt",
            "Johannesburg": "South Africa",
            "Cape Town": "South Africa",
            "Lagos": "Nigeria",
            "Nairobi": "Kenya",
            "Addis Ababa": "Ethiopia",
            "Casablanca": "Morocco",
            "Tunis": "Tunisia",
            "Algiers": "Algeria",
            "Dakar": "Senegal",

            # --- New locations added from methods_150_locations.json ---
            "Orange County": "United States",
            "Irvine": "United States",
            "Palmas": "Brazil",
            "Tocantins": "Brazil",
            "Porto Alegre": "Brazil",
            "Fars": "Iran",
            "Mazandaran": "Iran",
            "Nanjing": "China",
            "Durban": "South Africa",
            "the Democratic Republic of Congo": "Democratic Republic of Congo",
            "Congo": "Democratic Republic of Congo",
            "Katana": "Democratic Republic of Congo",
            "South Kivu": "Democratic Republic of Congo",
            "Dar es Salaam": "Tanzania",
            "Redmond": "United States",
            "Illubabor": "Ethiopia",
            "Oromia": "Ethiopia",
            "Western Australia": "Australia",
            "Rhode Island": "United States",
            "Segamat": "Malaysia",
            "Johor State": "Malaysia",
            "Belfast": "United Kingdom",
            "Kermanshah": "Iran",
            "Isfahan province": "Iran",
            "Shiraz": "Iran",
            "Marburg": "Germany",
            "Amadora": "Portugal",
            "West Midlands": "United Kingdom",
            "Bhaktapur district": "Nepal",
            "Lima": "Peru",
            "Klang Valley": "Malaysia",
            "Selangor": "Malaysia",
            "Uppsala county": "Sweden",
            "Loreto": "Peru",
            "Iquitos": "Peru",
            "Dunedin": "New Zealand",
            "Nouna": "Burkina Faso",
            "Kossi": "Burkina Faso",
            "Canberra": "Australia",
            "Domingo": "Dominican Republic",
            "Lalitpur": "Nepal",
            "Kano": "Nigeria",
            "Bayelsa": "Nigeria",
            "Port Harcourt": "Nigeria",
            "Sari": "Iran",
            "Ontario": "Canada",
            "Brighton": "United Kingdom",
            "Hove": "United Kingdom",
            "East Sussex": "United Kingdom",
            "Lothian": "United Kingdom",
            "Abeshige": "Ethiopia",
            "Townsville": "Australia",
            "Kitchener": "Canada",
            "Mississauga": "Canada",
            "(Southeast) Michigan": "United States",
            "the Province of Ontario": "Canada",
            "Midlands": "United Kingdom",
            "East Anglia": "United Kingdom",
            "Gloucestershire": "United Kingdom",
            "South Wales": "United Kingdom",
            "Burlington": "United States",
            "Gainesville": "United States",
            "Detroit": "United States",
            "Jackson": "United States",
            "Nashville": "United States",
            "New Orleans": "United States",
            "Oakland": "United States",
            "Omaha": "United States",
            "Basel": "Switzerland",
            "Zwolle": "Netherlands",
            "Lawrence": "United States"
        }
        return city_map
        
    def _get_country_continent_map(self):
        """
        Get mapping of countries to continents
        
        Returns:
            dict: Dictionary with countries as keys and continents as values
        """
        return {
            # North America
            "United States": "North America",
            "Canada": "North America",
            "Mexico": "North America",
            "Guatemala": "North America",
            "Cuba": "North America",
            "Haiti": "North America",
            "Jamaica": "North America",
            "Dominican Republic": "North America",
            "Puerto Rico": "North America",
            "Costa Rica": "North America",
            "Panama": "North America",
            
            # South America
            "Brazil": "South America",
            "Argentina": "South America",
            "Chile": "South America",
            "Peru": "South America",
            "Colombia": "South America",
            "Venezuela": "South America",
            "Ecuador": "South America",
            "Bolivia": "South America",
            "Paraguay": "South America",
            "Uruguay": "South America",
            
            # Europe
            "United Kingdom": "Europe",
            "France": "Europe",
            "Germany": "Europe",
            "Italy": "Europe",
            "Spain": "Europe",
            "Portugal": "Europe",
            "Netherlands": "Europe",
            "Belgium": "Europe",
            "Switzerland": "Europe",
            "Austria": "Europe",
            "Sweden": "Europe",
            "Norway": "Europe",
            "Denmark": "Europe",
            "Finland": "Europe",
            "Iceland": "Europe",
            "Greece": "Europe",
            "Turkey": "Europe",
            "Poland": "Europe",
            "Romania": "Europe",
            "Ukraine": "Europe",
            "Russia": "Europe",
            "Russian Federation": "Europe",
            "Hungary": "Europe",
            "Czech Republic": "Europe",
            "Slovakia": "Europe",
            "Slovenia": "Europe",
            "Croatia": "Europe",
            "Serbia": "Europe",
            "Bulgaria": "Europe",
            "Latvia": "Europe",
            "Lithuania": "Europe",
            "Estonia": "Europe",
            "Belarus": "Europe",
            "Moldova": "Europe",
            "Ireland": "Europe",
            
            # Asia
            "China": "Asia",
            "Japan": "Asia",
            "South Korea": "Asia",
            "North Korea": "Asia",
            "India": "Asia",
            "Pakistan": "Asia",
            "Bangladesh": "Asia",
            "Sri Lanka": "Asia",
            "Nepal": "Asia",
            "Bhutan": "Asia",
            "Thailand": "Asia",
            "Vietnam": "Asia",
            "Cambodia": "Asia",
            "Laos": "Asia",
            "Myanmar": "Asia",
            "Malaysia": "Asia",
            "Singapore": "Asia",
            "Indonesia": "Asia",
            "Philippines": "Asia",
            "Saudi Arabia": "Asia",
            "Iran": "Asia",
            "Iraq": "Asia",
            "Israel": "Asia",
            "Jordan": "Asia",
            "Lebanon": "Asia",
            "Syria": "Asia",
            "United Arab Emirates": "Asia",
            "Qatar": "Asia",
            "Kuwait": "Asia",
            "Bahrain": "Asia",
            "Oman": "Asia",
            "Yemen": "Asia",
            "Afghanistan": "Asia",
            "Kazakhstan": "Asia",
            "Uzbekistan": "Asia",
            "Tajikistan": "Asia",
            "Kyrgyzstan": "Asia",
            "Turkmenistan": "Asia",
            "Mongolia": "Asia",
            
            # Africa
            "Egypt": "Africa",
            "South Africa": "Africa",
            "Nigeria": "Africa",
            "Kenya": "Africa",
            "Ethiopia": "Africa",
            "Tanzania": "Africa",
            "Uganda": "Africa",
            "Ghana": "Africa",
            "Cameroon": "Africa",
            "Côte d'Ivoire": "Africa",
            "Ivory Coast": "Africa",
            "Senegal": "Africa",
            "Mali": "Africa",
            "Niger": "Africa",
            "Chad": "Africa",
            "Sudan": "Africa",
            "South Sudan": "Africa",
            "Morocco": "Africa",
            "Algeria": "Africa",
            "Tunisia": "Africa",
            "Libya": "Africa",
            "Angola": "Africa",
            "Zimbabwe": "Africa",
            "Zambia": "Africa",
            "Botswana": "Africa",
            "Namibia": "Africa",
            "Democratic Republic of Congo": "Africa",
            "Burkina Faso": "Africa",
            "Burundi": "Africa",
            
            # Oceania
            "Australia": "Oceania",
            "New Zealand": "Oceania",
            "Papua New Guinea": "Oceania",
            "Fiji": "Oceania",
            "Solomon Islands": "Oceania",
            "Vanuatu": "Oceania",
            "Samoa": "Oceania",
            "Tonga": "Oceania"
        }
    
    def map_location_to_continent(self, location):
        """
        Map a location to its continent
        
        Args:
            location (str): Location name
            
        Returns:
            str: Continent name
        """
        continent_map = self._get_country_continent_map()
        
        # Try direct lookup
        if location in continent_map:
            return continent_map[location]
        
        # Try synonym lookup
        synonyms = self._get_country_synonyms()
        if location in synonyms:
            normalized_country = synonyms[location]
            if normalized_country in continent_map:
                return continent_map[normalized_country]
                
        # If not found, check if it's a substring of any entry
        for country, continent in continent_map.items():
            if country in location or location in country:
                return continent
                
        return "Unknown"
    
    def create_location_dataframe(self, location_counts):
        """
        Create a DataFrame with location information
        
        Args:
            location_counts (dict): Dictionary with locations as keys and counts as values
            
        Returns:
            pd.DataFrame: DataFrame with location data
        """
        locations = []
        
        for location, count in location_counts.items():
            # Get continent for the location
            continent = self.map_location_to_continent(location)
            
            # Only include if location has a reasonable length and isn't just a number
            if len(location) > 2 and not location.isdigit():
                locations.append({
                    "Location": location,
                    "Count": count,
                    "Continent": continent
                })
        
        # Create DataFrame from the list of dictionaries
        location_df = pd.DataFrame(locations)
        
        # If we got any locations, sort by count
        if not location_df.empty:
            location_df = location_df.sort_values("Count", ascending=False)
            
        return location_df
    
    def create_choropleth_map(self, location_df):
        """
        Create a world map visualization with bubbles for each location
        
        Args:
            location_df (pd.DataFrame): DataFrame with location data
            
        Returns:
            plotly.graph_objects.Figure: Map figure with bubble markers
        """
        # Location coordinates - map countries to lat/long
        location_coords = {
            # North America
            "United States": (37.0902, -95.7129),
            "Canada": (56.1304, -106.3468),
            "Mexico": (23.6345, -102.5528),
            "Guatemala": (15.7835, -90.2308),
            "Cuba": (21.5218, -77.7812),
            "Haiti": (18.9712, -72.2852),
            "Jamaica": (18.1096, -77.2975),
            "Dominican Republic": (18.7357, -70.1627),
            "Puerto Rico": (18.2208, -66.5901),
            "Costa Rica": (9.7489, -83.7534),
            "Panama": (8.5380, -80.7821),
            
            # South America
            "Brazil": (-14.2350, -51.9253),
            "Argentina": (-38.4161, -63.6167),
            "Chile": (-35.6751, -71.5430),
            "Peru": (-9.1900, -75.0152),
            "Colombia": (4.5709, -74.2973),
            "Venezuela": (6.4238, -66.5897),
            "Ecuador": (-1.8312, -78.1834),
            "Bolivia": (-16.2902, -63.5887),
            "Paraguay": (-23.4425, -58.4438),
            "Uruguay": (-32.5228, -55.7658),
            
            # Europe
            "United Kingdom": (55.3781, -3.4360),
            "France": (46.2276, 2.2137),
            "Germany": (51.1657, 10.4515),
            "Italy": (41.8719, 12.5674),
            "Spain": (40.4637, -3.7492),
            "Portugal": (39.3999, -8.2245),
            "Netherlands": (52.1326, 5.2913),
            "Belgium": (50.5039, 4.4699),
            "Switzerland": (46.8182, 8.2275),
            "Austria": (47.5162, 14.5501),
            "Sweden": (60.1282, 18.6435),
            "Norway": (60.4720, 8.4689),
            "Denmark": (56.2639, 9.5018),
            "Finland": (61.9241, 25.7482),
            "Iceland": (64.9631, -19.0208),
            "Greece": (39.0742, 21.8243),
            "Turkey": (38.9637, 35.2433),
            "Poland": (51.9194, 19.1451),
            "Romania": (45.9432, 24.9668),
            "Ukraine": (48.3794, 31.1656),
            "Russia": (61.5240, 105.3188),
            "Russian Federation": (61.5240, 105.3188),
            "Hungary": (47.1625, 19.5033),
            "Czech Republic": (49.8175, 15.4730),
            "Slovakia": (48.6690, 19.6990),
            "Slovenia": (46.1512, 14.9955),
            "Croatia": (45.1000, 15.2000),
            "Serbia": (44.0165, 21.0059),
            "Bulgaria": (42.7339, 25.4858),
            "Latvia": (56.8796, 24.6032),
            "Lithuania": (55.1694, 23.8813),
            "Estonia": (58.5953, 25.0136),
            "Belarus": (53.7098, 27.9534),
            "Moldova": (47.4116, 28.3699),
            "Ireland": (53.1424, -7.6921),
            
            # Asia
            "China": (35.8617, 104.1954),
            "Japan": (36.2048, 138.2529),
            "South Korea": (35.9078, 127.7669),
            "North Korea": (40.3399, 127.5101),
            "India": (20.5937, 78.9629),
            "Pakistan": (30.3753, 69.3451),
            "Bangladesh": (23.6850, 90.3563),
            "Sri Lanka": (7.8731, 80.7718),
            "Nepal": (28.3949, 84.1240),
            "Bhutan": (27.5142, 90.4336),
            "Thailand": (15.8700, 100.9925),
            "Vietnam": (14.0583, 108.2772),
            "Cambodia": (12.5657, 104.9910),
            "Laos": (19.8563, 102.4955),
            "Myanmar": (21.9162, 95.9560),
            "Malaysia": (4.2105, 101.9758),
            "Singapore": (1.3521, 103.8198),
            "Indonesia": (-0.7893, 113.9213),
            "Philippines": (12.8797, 121.7740),
            "Saudi Arabia": (23.8859, 45.0792),
            "Iran": (32.4279, 53.6880),
            "Iraq": (33.2232, 43.6793),
            "Israel": (31.0461, 34.8516),
            "Jordan": (30.5852, 36.2384),
            "Lebanon": (33.8547, 35.8623),
            "Syria": (34.8021, 38.9968),
            "United Arab Emirates": (23.4241, 53.8478),
            "Qatar": (25.3548, 51.1839),
            "Kuwait": (29.3117, 47.4818),
            "Bahrain": (26.0667, 50.5577),
            "Oman": (21.4735, 55.9754),
            "Yemen": (15.5527, 48.5164),
            "Afghanistan": (33.9391, 67.7100),
            "Kazakhstan": (48.0196, 66.9237),
            "Uzbekistan": (41.3775, 64.5853),
            "Tajikistan": (38.8610, 71.2761),
            "Kyrgyzstan": (41.2044, 74.7661),
            "Turkmenistan": (38.9697, 59.5563),
            "Mongolia": (46.8625, 103.8467),
            
            # Africa
            "Egypt": (26.8206, 30.8025),
            "South Africa": (-30.5595, 22.9375),
            "Nigeria": (9.0820, 8.6753),
            "Kenya": (-0.0236, 37.9062),
            "Ethiopia": (9.1450, 40.4897),
            "Tanzania": (-6.3690, 34.8888),
            "Uganda": (1.3733, 32.2903),
            "Ghana": (7.9465, -1.0232),
            "Cameroon": (7.3697, 12.3547),
            "Côte d'Ivoire": (7.5400, -5.5471),
            "Ivory Coast": (7.5400, -5.5471),
            "Senegal": (14.4974, -14.4524),
            "Mali": (17.5707, -3.9962),
            "Niger": (17.6078, 8.0817),
            "Chad": (15.4542, 18.7322),
            "Sudan": (12.8628, 30.2176),
            "South Sudan": (6.8770, 31.3070),
            "Morocco": (31.7917, -7.0926),
            "Algeria": (28.0339, 1.6596),
            "Tunisia": (33.8869, 9.5375),
            "Libya": (26.3351, 17.2283),
            "Angola": (-11.2027, 17.8739),
            "Zimbabwe": (-19.0154, 29.1549),
            "Zambia": (-13.1339, 27.8493),
            "Botswana": (-22.3285, 24.6849),
            "Namibia": (-22.9576, 18.4904),
            "Democratic Republic of Congo": (-4.0383, 21.7587),
            "Burkina Faso": (12.2383, -1.5616),
            "Burundi": (-3.3731, 29.9189),

            # Oceania
            "Australia": (-25.2744, 133.7751),
            "New Zealand": (-40.9006, 174.8860),
            "Papua New Guinea": (-6.3150, 143.9555),
            "Fiji": (-17.7134, 178.0650),
            "Solomon Islands": (-9.6457, 160.1562),
            "Vanuatu": (-15.3767, 166.9592),
            "Samoa": (-13.7590, -172.1046),
            "Tonga": (-21.1790, -175.1982)
        }
        
        # Filter locations to those we have coordinates for
        locations_with_coords = []
        for index, row in location_df.iterrows():
            location = row['Location']
            count = row['Count']
            continent = row['Continent']
            
            if location in location_coords:
                lat, lon = location_coords[location]
                locations_with_coords.append({
                    'Location': location,
                    'Count': count,
                    'Continent': continent,
                    'lat': lat,
                    'lon': lon
                })
        
        # If no coordinates found, return an empty map with message
        if not locations_with_coords:
            fig = go.Figure()
            fig.add_annotation(
                text="No mappable locations found",
                xref="paper", yref="paper",
                x=0.5, y=0.5,
                showarrow=False,
                font=dict(size=20)
            )
            return fig
            
        # Create dataframe from the filtered locations
        map_df = pd.DataFrame(locations_with_coords)
        
        # Define a color map for continents
        continent_colors = {
            "North America": "#1f77b4",  # blue
            "South America": "#ff7f0e",  # orange
            "Europe": "#2ca02c",         # green
            "Asia": "#d62728",           # red
            "Africa": "#9467bd",         # purple
            "Oceania": "#8c564b",        # brown
            "Unknown": "#7f7f7f"         # gray
        }
        
        map_df['color'] = map_df['Continent'].map(continent_colors)
        
        # Create the map
        fig = go.Figure()
        
        # Add each location as a bubble
        for continent in map_df['Continent'].unique():
            continent_df = map_df[map_df['Continent'] == continent]
            
            fig.add_trace(go.Scattergeo(
                lon=continent_df['lon'],
                lat=continent_df['lat'],
                text=continent_df.apply(lambda row: f"{row['Location']}<br>Articles: {row['Count']}", axis=1),
                marker=dict(
                    size=continent_df['Count'] * 5,  # Size proportional to count
                    color=continent_colors.get(continent, "#7f7f7f"),
                    line_width=0,
                    opacity=0.7,
                    sizemode='area',
                    sizemin=5
                ),
                name=continent,
                hoverinfo='text',
                showlegend=True
            ))
        
        # Update layout for better appearance
        fig.update_layout(
            title="Geographic Distribution of Location Mentions",
            geo=dict(
                showland=True,
                landcolor='rgb(243, 243, 243)',
                countrycolor='rgb(204, 204, 204)',
                showocean=True,
                oceancolor='rgb(230, 230, 255)',
                showframe=False,
                showcountries=True,
                projection_type='natural earth',
                lonaxis=dict(range=[-180, 180]),
                lataxis=dict(range=[-90, 90]),
                showcoastlines=True
            ),
            margin=dict(l=0, r=0, t=40, b=0),
            legend=dict(
                orientation="h",
                y=0,
                x=0.5,
                xanchor="center"
            )
        )
        
        return fig
        
    def find_articles_with_location(self, pre_extracted_data, target_country):
        """
        Find articles that mention a specific location (after normalization).
        
        Args:
            pre_extracted_data (list): List of article dictionaries from the locations JSON.
            target_country (str): The *normalized* country name to search for.
            
        Returns:
            list: List of article dictionaries mentioning the location.
        """
        matches = []
        
        for article in pre_extracted_data:
            raw_locations = article.get("extracted_locations", [])
            if not raw_locations:
                continue
                
            found = False
            for loc in raw_locations:
                # Normalize the location
                country = self._get_country_for_location(loc, self.city_to_country_map)
                if country in self.country_synonyms:
                    country = self.country_synonyms[country]
                    
                if country == target_country:
                    found = True
                    break
            
            if found:
                matches.append({
                    "title": article.get("article_title", article.get("title", "No Title")),
                    "pmid": article.get("pmid", "No PMID"),
                    "doi": article.get("doi", "No DOI"),
                    "year": article.get("year", "Unknown")
                })
        
        return matches