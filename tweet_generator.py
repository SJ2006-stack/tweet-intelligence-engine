import random

class SimpleTweetGenerator:
    def __init__(self):
        # Expanded templates with more types and varied styles
        self.templates = {
            'announcement': [
                "🚀 Exciting news from {company}! {message}",
                "Big announcement: {company} is {message} 🎉",
                "Hey everyone! {company} has {message} ✨"
            ],
            'question': [
                "What do you think about {topic}? Let us know! 💬",
                "Quick question: How do you feel about {topic}? 🤔",
                "{company} wants to know: What's your take on {topic}? 🗣️"
            ],
            'general': [
                "Check out what {company} is up to! {message} 🌟",
                "{company} update: {message} 💯",
                "From the {company} team: {message} 🔥"
            ],
            'celebration': [
                "🎉 Celebrating a huge milestone at {company}: {message}! Thank you all for the support! 🥳",
                "We did it! {company} just {message} 🎈 #Milestone",
                "{company} is thrilled to announce: {message} 🎊"
            ],
            'tip': [
                "💡 Tip of the day from {company}: {message}",
                "Here’s a quick tip on {topic} from {company} to boost your day! 🚀",
                "{company} suggests: {message} #TipsAndTricks"
            ],
            'quote': [
                "“{message}” - {company}",
                "{company} shares: \"{message}\" #QuoteOfTheDay",
                "Words to live by from {company}: \"{message}\" ✨"
            ],
            'event': [
                "📅 Don't miss {company}'s upcoming event: {message} Join us!",
                "Save the date! {company} is hosting {message} 📢",
                "{company} invites you to {message} 🔔"
            ],
            'product_launch': [
                "Introducing our latest product: {message} by {company}! 🚀",
                "Say hello to {message} from {company} 🌟 #NewRelease",
                "{company} proudly launches: {message} 🔥"
            ]
        }

    def generate_tweet(self, company, tweet_type="general", message=None, topic=None):
        # Validate company name
        if not company or company.strip() == "":
            return "Error: Company name is required."

        tpltype = (tweet_type or "general").strip().lower()
        # Use 'general' if tweet_type is not recognized
        if tpltype not in self.templates:
            tpltype = 'general'

        # Assign message and topic defaults per tweet type if missing
        if not message or message.strip() == "":
            if tpltype == 'announcement':
                message = "making big moves"
            elif tpltype == 'question':
                message = ""
            elif tpltype == 'celebration':
                message = "reaching new heights"
            elif tpltype == 'tip':
                message = "stay positive and keep learning"
            elif tpltype == 'quote':
                message = "Success is not final; failure is not fatal: It is the courage to continue that counts."
            elif tpltype == 'event':
                message = "our annual meetup next week"
            elif tpltype == 'product_launch':
                message = "the latest innovation in tech"
            else:
                message = "something awesome!"

        if not topic or topic.strip() == "":
            if tpltype == 'question':
                topic = "this topic"
            elif tpltype == 'tip':
                topic = "daily life"
            else:
                topic = "innovation"

        template_list = self.templates[tpltype]
        if not template_list:
            return "Error: No templates found for this tweet type."

        template = random.choice(template_list)

        try:
            tweet = template.format(
                company=company,
                message=message,
                topic=topic
            )
        except Exception as e:
            return f"Error: failed to create tweet ({e})."

        # Truncate if too long (max 280 tweets)
        if len(tweet) > 280:
            tweet = tweet[:277] + "..."

        return tweet

    def get_supported_tweet_types(self):
        """Returns the list of supported tweet types"""
        return list(self.templates.keys())

# Example usage and testing
if __name__ == "__main__":
    generator = SimpleTweetGenerator()

    test_cases = [
        {"company": "Nike", "tweet_type": "announcement", "message": "launching new running shoes"},
        {"company": "Starbucks", "tweet_type": "question", "topic": "coffee"},
        {"company": "Apple", "tweet_type": "celebration", "message": "hitting $1B in sales this quarter"},
        {"company": "Tesla", "tweet_type": "general", "message": "changing the world"},
        {"company": "Microsoft", "tweet_type": "tip", "message": "Always backup your files"},
        {"company": "SpaceX", "tweet_type": "quote", "message": "To infinity and beyond!"},
        {"company": "Google", "tweet_type": "event", "message": "launching Google I/O 2025"},
        {"company": "Samsung", "tweet_type": "product_launch", "message": "Galaxy Z Fold 5"},
        {"company": "   ", "tweet_type": "general", "message": "missing company"},  # Invalid company
        {"company": "Unknown", "tweet_type": "unknown", "message": "testing unknown tweet_type"},
    ]

    for i, case in enumerate(test_cases, 1):
        tweet = generator.generate_tweet(
            company=case.get("company", ""),
            tweet_type=case.get("tweet_type", "general"),
            message=case.get("message"),
            topic=case.get("topic")
        )
        print(f"Test {i}:", tweet)

    print("\nSupported tweet types:", generator.get_supported_tweet_types())
