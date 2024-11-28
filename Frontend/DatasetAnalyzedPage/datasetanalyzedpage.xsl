<?xml version="1.0" encoding="UTF-8"?>
<xsl:stylesheet version="1.0" xmlns:xsl="http://www.w3.org/1999/XSL/Transform">
    <xsl:template match="/">
        <html>
            <head>
                <title>Model Statistics</title>
                <style>
                    table { width: 100%; border-collapse: collapse; }
                    th, td { padding: 10px; text-align: left; border: 1px solid #ddd; }
                    th { background-color: #f2f2f2; }
                    h1, h2 { text-align: center; }
                </style>
            </head>
                <body>
                <h1>Analysis of Dataset</h1>
                <h3>Pages:</h3>
                <ul>
                    <li><button onclick="window.location.href='../MainPage/mainpage.html'">Main Page</button></li>
                    <li><button onclick="window.location.href='../DatasetAnalyzedPage/df_analyzed.xml'">DataFrame Analyzed</button></li>
                    <li><button onclick="window.location.href='../FeaturesOfModelsPage/Log_model_scores_df.xml'">Features Of Models</button></li>
                </ul>
                <!-- Analysis of Dataset -->
                <Table border="1">
                    <tr bgcolor="#aaaaaa">
                        <th>Comment Sentiment type</th>
                        <th>Quantities of negative/positive comments</th>
                        <th>Overall rating for comment polarity</th>
                        <th>Number of Helpfulness Votes</th>
                    </tr>
                    <xsl:for-each select="document('df_analyzed.xml')/data/row">
                        <tr>
                            <td><xsl:value-of select="sentiment_label"/></td>
                            <td><xsl:value-of select="count"/></td>
                            <td><xsl:value-of select="Star"/></td>
                            <td><xsl:value-of select="HelpFul"/></td>
                        </tr>
                    </xsl:for-each>
                </Table>
            </body>
        </html>
    </xsl:template>
</xsl:stylesheet>